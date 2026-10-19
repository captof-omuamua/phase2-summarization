"""Train the custom Malayalam Transformer on frozen local publisher splits."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

# Set before importing/initializing CUDA; recorded in the run protocol.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import TransformerSummarizer
from preprocess import PAD, BOS, EOS, file_hash, read_jsonl, verify_prepared, write_json

ROOT = Path(__file__).resolve().parent
MODEL_SETTINGS = dict(d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                      dim_feedforward=512, dropout=0.1, bos_id=BOS, eos_id=EOS, pad_id=PAD)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def collate_examples(rows):
    def padded(name, value):
        sequences = [torch.tensor(row[name], dtype=torch.long) for row in rows]
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=value)
    return {"source": padded("source", PAD), "decoder_input": padded("decoder_input", PAD),
            "labels": padded("labels", -100)}


def forward_loss(model, batch):
    source, decoder, labels = batch["source"], batch["decoder_input"], batch["labels"]
    source_mask, target_padding = source.eq(PAD), decoder.eq(PAD)
    logits = model(source, decoder, tgt_mask=model._generate_square_subsequent_mask(decoder.size(1)).to(source.device),
                   src_key_padding_mask=source_mask, tgt_key_padding_mask=target_padding,
                   memory_key_padding_mask=source_mask)
    total = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
                            ignore_index=-100, reduction="sum")
    tokens = int(labels.ne(-100).sum().item())
    if not tokens or not torch.isfinite(total):
        raise FloatingPointError("Nonfinite loss or no supervised target tokens")
    return total, tokens


def precision_context(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bfloat16" and device.type == "cuda" else nullcontext()


def run_epoch(model, loader, device, precision, optimizer=None, max_steps=None):
    model.train(optimizer is not None)
    total_loss, total_tokens, batches = 0.0, 0, 0
    start = time.perf_counter()
    for index, batch in enumerate(loader):
        if max_steps is not None and index >= max_steps:
            break
        batch = {name: value.to(device) for name, value in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with precision_context(device, precision):
                loss_sum, tokens = forward_loss(model, batch)
                loss = loss_sum / tokens
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
        total_loss += float(loss_sum.detach().item())
        total_tokens += tokens
        batches += 1
    if not total_tokens:
        raise ValueError("Empty training/validation loader")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"token_cross_entropy": total_loss / total_tokens, "loss_sum": total_loss,
            "target_tokens": total_tokens, "batches": batches, "seconds": time.perf_counter() - start}


def load_model(checkpoint, prepared_dir, device):
    manifest = verify_prepared(prepared_dir)
    protocol = verify_run_sources(Path(checkpoint).parent)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["prepared_manifest_sha256"] != file_hash(Path(prepared_dir) / "prepared_manifest.json"):
        raise ValueError("Checkpoint and prepared dataset/tokenizer/splits do not match")
    if state["protocol_sha256"] != file_hash(Path(checkpoint).parent / "protocol.json") or state["model_config"] != protocol["model_config"]:
        raise ValueError("Checkpoint and frozen protocol/model configuration do not match")
    model = TransformerSummarizer(**state["model_config"]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, state, manifest


def verify_run_sources(run_dir):
    protocol = json.loads((Path(run_dir) / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["source_sha256"].items():
        if file_hash(ROOT / name) != expected or file_hash(Path(run_dir) / "source" / name) != expected:
            raise ValueError(f"Run source snapshot/current code hash mismatch: {name}")
    return protocol


def runtime_info(device, precision):
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": precision, "deterministic_algorithms": "enabled, warn_only=False",
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "reproducibility": "Recorded seeds/environment; no bitwise guarantee across hardware/library versions"}


def train(prepared_dir, output_dir, device_name="cuda", precision="bfloat16", batch_size=32,
          epochs=20, pilot_steps=None, resume=None):
    prepared, output = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not resume:
        raise ValueError("Training output directory must be fresh (or explicitly resume)")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("bfloat16 requires a supported CUDA GPU; explicitly select float32 otherwise")
    torch.set_num_threads(4)
    seed_everything()
    manifest = verify_prepared(prepared)
    config = {**MODEL_SETTINGS, "vocab_size": manifest["vocab_size"]}
    model = TransformerSummarizer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    generator = torch.Generator().manual_seed(42)
    train_rows, dev_rows = read_jsonl(prepared / "train_tokens.jsonl"), read_jsonl(prepared / "dev_tokens.jsonl")
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, generator=generator,
                              collate_fn=collate_examples, num_workers=0)
    dev_loader = DataLoader(dev_rows, batch_size=batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "prepared_manifest_sha256": file_hash(prepared / "prepared_manifest.json"),
        "model_config": config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_rows": len(train_rows), "dev_rows": len(dev_rows), "batch_size": batch_size,
        "max_epochs": epochs, "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
        "gradient_clip_norm": 1.0, "early_stopping": {"patience": 3, "min_improvement": 1e-4},
        "checkpoint_selection": "lowest validation token-weighted cross-entropy (never test metrics)",
        "scheduler": {"kind": "ReduceLROnPlateau", "factor": 0.5, "patience": 1},
        "decoding": {"kind": "batched greedy", "max_new_tokens": 64, "per_example_EOS": True},
        "metrics": {"rouge": "macro F1; Unicode L/M/N runs; NFC; no stemming; scale0-100",
                    "chrF": {"char_order": 6, "word_order": 0, "beta": 2, "whitespace": False}},
        "lead_words": manifest["lead_words"], "environment": runtime_info(device, precision),
        "pilot_steps": pilot_steps, "source_sha256": {},
    }
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for path in sorted(ROOT.glob("*.py")) + [ROOT / "requirements.in", ROOT / ".gitattributes"]:
        if path.is_file():
            if not resume:
                shutil.copyfile(path, source_dir / path.name)
            protocol["source_sha256"][path.name] = file_hash(path)
    start_epoch, best_loss, early_best, no_improve, history = 1, math.inf, math.inf, 0, []
    best_epoch = None
    if resume:
        previous = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        for field in ["prepared_manifest_sha256", "model_config", "batch_size", "max_epochs", "source_sha256", "environment"]:
            if previous[field] != protocol[field]:
                raise ValueError(f"Resume protocol mismatch: {field}")
        protocol = previous
        saved = torch.load(resume, map_location=device, weights_only=True)
        if (saved["prepared_manifest_sha256"] != previous["prepared_manifest_sha256"]
                or saved["model_config"] != config
                or saved["protocol_sha256"] != file_hash(output / "protocol.json")):
            raise ValueError("Resume checkpoint identity does not match frozen run protocol")
        if saved["no_improve"] >= 3 or saved["epoch"] >= previous["max_epochs"]:
            raise ValueError("Cannot extend a completed or early-stopped frozen training run")
        verify_run_sources(output)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch, best_loss, best_epoch = saved["epoch"] + 1, saved["best_loss"], saved["best_epoch"]
        early_best, no_improve, history = saved["early_best"], saved["no_improve"], saved["history"]
        random.setstate(saved["python_rng"])
        nprng = saved["numpy_rng"]
        np.random.set_state((nprng[0], np.array(nprng[1], dtype=np.uint32), *nprng[2:]))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        generator.set_state(saved["loader_rng"].cpu())
    else:
        write_json(output / "protocol.json", protocol)
        shutil.copyfile(prepared / "prepared_manifest.json", output / "prepared_manifest.json")
        shutil.copyfile(prepared / "tokenizer.model", output / "tokenizer.model")
    write_json(output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    if pilot_steps:
        result = run_epoch(model, train_loader, device, precision, optimizer, max_steps=pilot_steps)
        result.update(pilot_only=True, optimizer_updates=result["batches"], parameter_count=protocol["parameter_count"],
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        write_json(output / "pilot.json", result)
        print(json.dumps(result, indent=2))
        write_json(output / "run_status.json", {"status": "pilot_complete", "weights_discarded": True})
        return result
    try:
        for epoch in range(start_epoch, epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            training = run_epoch(model, train_loader, device, precision, optimizer)
            validation = run_epoch(model, dev_loader, device, precision)
            val_loss = validation["token_cross_entropy"]
            history.append({"epoch": epoch, "learning_rate": learning_rate, "training": training, "validation": validation})
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                            "prepared_manifest_sha256": protocol["prepared_manifest_sha256"],
                            "protocol_sha256": file_hash(output / "protocol.json"),
                            "epoch": epoch, "validation_loss": val_loss, "seed": 42}, output / "best.pt")
            if val_loss < early_best - 1e-4:
                early_best, no_improve = val_loss, 0
            else:
                no_improve += 1
            scheduler.step(val_loss)
            nprng = np.random.get_state()
            torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "prepared_manifest_sha256": protocol["prepared_manifest_sha256"], "epoch": epoch,
                        "protocol_sha256": file_hash(output / "protocol.json"),
                        "best_loss": best_loss, "best_epoch": best_epoch, "early_best": early_best,
                        "no_improve": no_improve, "history": history, "python_rng": random.getstate(),
                        "numpy_rng": (nprng[0], nprng[1].tolist(), *nprng[2:]),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "loader_rng": generator.get_state()}, output / "last.pt")
            write_json(output / "history.json", history)
            print(f"Epoch{epoch}: trainNLL={training['token_cross_entropy']:.6f} valNLL={val_loss:.6f} train={training['seconds']:.1f}s val={validation['seconds']:.1f}s", flush=True)
            if no_improve >= 3:
                break
        selection = {"selected_epoch": best_epoch, "validation_token_cross_entropy": best_loss,
                     "checkpoint_sha256": file_hash(output / "best.pt"),
                     "protocol_sha256": file_hash(output / "protocol.json"),
                     "selected_before_test_utc": datetime.now(timezone.utc).isoformat(),
                     "training_seconds": time.perf_counter() - start, "test_used": False}
        write_json(output / "selection.json", selection)
        write_json(output / "run_status.json", {"status": "training_complete", **selection})
        return selection
    except Exception as exc:
        write_json(output / "run_status.json", {"status": "failed", "error": str(exc), "error_type": type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.prepared_dir, args.output_dir, args.device, args.precision, args.batch_size,
          args.epochs, args.pilot_steps, args.resume)


if __name__ == "__main__":
    main()
