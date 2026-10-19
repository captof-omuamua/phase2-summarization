"""Evaluate the frozen checkpoint and Lead-N baseline on the held-out test split."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import unicodedata

import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF
import sacrebleu

from preprocess import EOS, Tokenizer, file_hash, read_jsonl, write_json, write_jsonl
from train import collate_examples, load_model, precision_context, run_epoch, seed_everything


class UnicodeTokenizer:
    """NFC runs of Unicode letters, combining marks and numbers; case-sensitive."""
    def tokenize(self, text):
        tokens, current = [], []
        for character in unicodedata.normalize("NFC", text):
            if unicodedata.category(character)[0] in "LMN":
                current.append(character)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens


def score_predictions(predictions):
    """Same rule for both systems; all rows count, including empty predictions."""
    if not predictions:
        raise ValueError("No predictions to score")
    if len({row["identity"] for row in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction identities")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=UnicodeTokenizer())
    chrf = CHRF(char_order=6, word_order=0, beta=2, whitespace=False)
    aggregate, individual = {}, []
    for method in ["transformer", "lead_baseline"]:
        records = []
        for row in predictions:
            scores = scorer.score(row["reference"], row[method])
            values = {metric: float(scores[metric].fmeasure * 100) for metric in scores}
            individual.append({"identity": row["identity"], "method": method, **values})
            records.append(values)
        corpus = chrf.corpus_score([row[method] for row in predictions], [[row["reference"] for row in predictions]])
        aggregate[method] = {"examples": len(predictions),
                             **{metric: statistics.mean(row[metric] for row in records) for metric in records[0]},
                             "chrF": float(corpus.score),
                             "empty_prediction_rate": sum(not row[method].strip() for row in predictions) / len(predictions)}
    return {"systems": aggregate, "metric_scale": "0-100", "rouge": "macro F1, NFC Unicode L/M/N runs, case-sensitive, no stemming",
            "chrF_signature": str(chrf.get_signature()), "sacrebleu_version": sacrebleu.__version__}, individual


def recompute(prediction_path, output_path):
    """Re-read persisted generated strings and recompute all aggregate overlaps."""
    result, _ = score_predictions(read_jsonl(prediction_path))
    write_json(output_path, result)
    return result


def evaluate(prepared_dir, run_dir, output_dir, device_name="cuda", batch_size=32):
    prepared, run, output = Path(prepared_dir), Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Evaluation output directory must be fresh")
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if selection["checkpoint_sha256"] != file_hash(run / "best.pt"):
        raise ValueError("Selected checkpoint hash mismatch")
    if selection["protocol_sha256"] != file_hash(run / "protocol.json"):
        raise ValueError("Selected protocol hash mismatch")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA GPU is unavailable")
    seed_everything()
    torch.set_num_threads(4)
    model, state, manifest = load_model(run / "best.pt", prepared, device)
    if state["epoch"] != selection["selected_epoch"] or state["protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Checkpoint selection metadata mismatch")
    if protocol["decoding"] != {"kind": "batched greedy", "max_new_tokens": 64, "per_example_EOS": True}:
        raise ValueError("Unsupported or changed frozen decode protocol")
    tokenizer = Tokenizer(prepared / "tokenizer.model")
    sources = read_jsonl(prepared / "test.jsonl")
    tokens = read_jsonl(prepared / "test_tokens.jsonl")
    if [r["identity"] for r in sources] != [r["identity"] for r in tokens]:
        raise ValueError("Test source/token identities do not match")
    precision = protocol["environment"]["precision"]
    if precision == "bfloat16" and device.type != "cuda":
        raise ValueError("Evaluation must preserve the frozen CUDA bfloat16 precision")
    started = time.perf_counter()
    predictions = []
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "selection_sha256": file_hash(run / "selection.json"), "test_rows": len(sources)}
    write_json(output / "evaluation_status.json", status)
    try:
        model.eval()
        for start in range(0, len(tokens), batch_size):
            batch_tokens = tokens[start:start + batch_size]
            # Generation receives source IDs only; reference text never enters decoding.
            batch = collate_examples(batch_tokens)
            source_ids = batch["source"].to(device)
            with precision_context(device, precision):
                generated = model.generate_ids(source_ids, max_length=64).cpu().tolist()
            for row, ids in zip(sources[start:start + batch_size], generated):
                emitted = ids[1:]
                terminated = EOS in emitted
                content_ids = emitted[:emitted.index(EOS)] if terminated else emitted
                predictions.append({"identity": row["identity"], "source_id": row["source_id"],
                                    "article_sha256": row["article_sha256"], "url": row["url"],
                                    "reference": row["headline"], "transformer": tokenizer.decode(content_ids),
                                    "lead_baseline": " ".join(row["article"].split()[:manifest["lead_words"]]),
                                    "generated_tokens": len(content_ids), "terminated_with_EOS": terminated})
            print(f"Decoded {min(start + batch_size, len(tokens))}/{len(tokens)} held-out examples", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        write_jsonl(output / "test_predictions.jsonl", predictions)
        scores, individual = score_predictions(predictions)
        write_jsonl(output / "per_example_metrics.jsonl", individual)
        reloaded, _ = score_predictions(read_jsonl(output / "test_predictions.jsonl"))
        if scores != reloaded:
            raise AssertionError("Persisted prediction metric recomputation differs")
        write_json(output / "recomputed_metrics.json", reloaded)
        loader = DataLoader(tokens, batch_size=batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0)
        teacher_forced = run_epoch(model, loader, device, precision)
        lengths = [row["generated_tokens"] for row in predictions]
        scores.update({"test_teacher_forced": teacher_forced, "selected_epoch": selection["selected_epoch"],
                       "generation_seconds": generation_seconds, "total_evaluation_seconds": time.perf_counter() - started,
                       "EOS_termination_rate": sum(row["terminated_with_EOS"] for row in predictions) / len(predictions),
                       "generated_token_lengths": {"min": min(lengths), "mean": statistics.mean(lengths),
                                                   "median": statistics.median(lengths), "max": max(lengths)},
                       "checkpoint_sha256": selection["checkpoint_sha256"], "selection_sha256": file_hash(run / "selection.json"),
                       "prepared_manifest_sha256": file_hash(prepared / "prepared_manifest.json"),
                       "prediction_file_sha256": file_hash(output / "test_predictions.jsonl"),
                       "lead_words": manifest["lead_words"], "truncation": manifest["truncation"],
                       "dataset_variant": manifest["dataset"], "license": manifest["license"]})
        write_json(output / "metrics.json", scores)
        # Stable identity order was fixed during preparation, before generation.
        write_jsonl(output / "deterministic_examples.jsonl", predictions[:5])
        status.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                      prediction_file_sha256=scores["prediction_file_sha256"])
        write_json(output / "evaluation_status.json", status)
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        return scores
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "evaluation_status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recompute-predictions", type=Path)
    args = parser.parse_args()
    if args.recompute_predictions:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        recompute(args.recompute_predictions, args.output_dir / "recomputed_metrics.json")
    else:
        if args.prepared_dir is None or args.run_dir is None:
            parser.error("--prepared-dir and --run-dir are required for model evaluation")
        evaluate(args.prepared_dir, args.run_dir, args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
