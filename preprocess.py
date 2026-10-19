"""Prepare immutable, deduplicated publisher splits and a training-only tokenizer."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import zipfile

ARCHIVE_SHA256 = "1c371ca6a152238aa6419cb334e70f58d36cd31a1a7a12ef59909ada243fcaa6"
REVISION = "d9845634dc0f9cb48d4a26c9f6d8986fb87d2027"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SOURCE_CAP, TARGET_CAP = 256, 64


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize_text(value):
    if not isinstance(value, str):
        raise ValueError("Article, title and URL fields must be strings")
    value.encode("utf-8", errors="strict")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def canonical_url(value):
    value = normalize_text(value)
    if not value:
        return ""
    parts = urlsplit(value)
    if not parts.hostname or parts.scheme.lower() not in {"http", "https"} or parts.username:
        raise ValueError("Nonempty source URLs must be ordinary http(s) URLs")
    host = parts.hostname.lower()
    if parts.port:
        host += f":{parts.port}"
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), host, parts.path, urlencode(sorted(query)), ""))


def deduplicate(raw_splits):
    """Union article/URL links transitively BEFORE test>dev>train precedence."""
    rows = []
    for split in ["train", "dev", "test"]:
        ids = set()
        for raw in raw_splits[split]:
            source_id = str(raw["id"])
            if not source_id or source_id in ids:
                raise ValueError(f"Missing/duplicate id within {split}: {source_id}")
            ids.add(source_id)
            article = normalize_text(raw["Document"])
            headline = normalize_text(raw["Title"])
            url = canonical_url(raw["URL"])
            if not article or not headline:
                raise ValueError(f"Empty article/headline at {split}:{source_id}")
            rows.append(dict(identity=f"{split}:{source_id}", source_id=source_id, split=split,
                             article=article, headline=headline, url=url,
                             article_sha256=digest(article), headline_sha256=digest(headline)))
    parents = list(range(len(rows)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    links = {}
    for index, row in enumerate(rows):
        keys = [("article", row["article_sha256"])]
        if row["url"]:
            keys.append(("url", row["url"]))
        for key in keys:
            if key in links:
                parents[find(index)] = find(links[key])
            else:
                links[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(rows[index])
    priority = {"test": 0, "dev": 1, "train": 2}
    kept = {name: [] for name in priority}
    manifest = []
    conflicting_titles = 0
    for component in groups.values():
        ordered = sorted(component, key=lambda row: (priority[row["split"]], row["identity"]))
        winner = ordered[0]
        component_id = digest("\n".join(sorted(row["identity"] for row in component)))
        conflicting_titles += len({row["headline_sha256"] for row in component}) > 1
        kept[winner["split"]].append(winner)
        for row in component:
            manifest.append({
                "identity": row["identity"], "split": row["split"], "source_id": row["source_id"],
                "article_sha256": row["article_sha256"], "headline_sha256": row["headline_sha256"],
                "canonical_url_sha256": digest(row["url"]) if row["url"] else None,
                "component": component_id, "representative": winner["identity"],
                "retained": row is winner,
                "reason": "retained" if row is winner else "duplicate_lower_priority_split" if row["split"] != winner["split"] else "duplicate_within_split",
            })
    for split in kept:
        kept[split].sort(key=lambda row: row["identity"])
        if not kept[split]:
            raise ValueError(f"Deduplication left {split} empty")
    used_articles, used_urls = set(), set()
    for split in ["test", "dev", "train"]:
        for row in kept[split]:
            assert row["article_sha256"] not in used_articles
            assert not row["url"] or row["url"] not in used_urls
            used_articles.add(row["article_sha256"])
            if row["url"]:
                used_urls.add(row["url"])
    audits = {}
    for split in ["train", "dev", "test"]:
        original = [row for row in rows if row["split"] == split]
        words = sorted(len(row["article"].split()) for row in original)
        target_words = sorted(len(row["headline"].split()) for row in original)
        language_count = sum(unicodedata.category(c)[0] in "LMN" for row in original for c in row["article"])
        malayalam_count = sum("\u0d00" <= c <= "\u0d7f" for row in original for c in row["article"]
                             if unicodedata.category(c)[0] in "LMN")
        audits[split] = {
            "raw_rows": len(original), "retained_rows": len(kept[split]), "excluded_rows": len(original) - len(kept[split]),
            "duplicate_article_occurrences": len(original) - len({row["article_sha256"] for row in original}),
            "duplicate_pair_occurrences": len(original) - len({(row["article_sha256"], row["headline_sha256"]) for row in original}),
            "duplicate_nonempty_url_occurrences": sum(bool(row["url"]) for row in original) - len({row["url"] for row in original if row["url"]}),
            "article_words": {"min": min(words), "median": statistics.median(words), "max": max(words)},
            "headline_words": {"min": min(target_words), "median": statistics.median(target_words), "max": max(target_words)},
            "malayalam_letter_mark_number_fraction": malayalam_count / max(1, language_count),
            "headline_verbatim_in_article": sum(row["headline"] in row["article"] for row in original),
        }
    overlaps = {}
    for first, second in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        one, two = [r for r in rows if r["split"] == first], [r for r in rows if r["split"] == second]
        overlaps[f"{first}/{second}"] = {
            "article_hashes": len({r["article_sha256"] for r in one} & {r["article_sha256"] for r in two}),
            "nonempty_urls": len({r["url"] for r in one if r["url"]} & {r["url"] for r in two if r["url"]}),
            "headline_hashes_only_descriptive": len({r["headline_sha256"] for r in one} & {r["headline_sha256"] for r in two}),
        }
    audit = {"splits": audits, "raw_cross_split_overlap": overlaps,
             "components_with_differing_titles": conflicting_titles,
             "retained_article_and_nonempty_url_overlap": 0,
             "deduplication": "transitive article OR canonical URL components; test>dev>train; lowest composite identity",
             "normalization": "Unicode NFC and whitespace only; no language/length/content filtering",
             "limitation": "Exact identity cleanup does not establish event-level or near-duplicate independence"}
    return kept, sorted(manifest, key=lambda r: r["identity"]), audit


class Tokenizer:
    def __init__(self, path):
        import sentencepiece as spm
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        if (self.processor.pad_id(), self.processor.unk_id(), self.processor.bos_id(), self.processor.eos_id()) != (PAD, UNK, BOS, EOS):
            raise ValueError("Tokenizer special IDs must be PAD0 UNK1 BOS2 EOS3")
        self.vocab_size = self.processor.get_piece_size()

    def encode(self, text):
        return self.processor.encode(text, out_type=int)

    def decode(self, tokens):
        tokens = list(tokens)
        if EOS in tokens:
            tokens = tokens[:tokens.index(EOS)]
        return self.processor.decode([token for token in tokens if token not in {PAD, BOS, EOS}])

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row) for row in rows]


def verify_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "prepared_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Prepared artifact hash mismatch: {name}")
    return manifest


def encode_example(article, headline, tokenizer, source_cap=SOURCE_CAP, target_cap=TARGET_CAP):
    if source_cap < 1 or target_cap < 2:
        raise ValueError("Source cap must be positive; target cap must reserve BOS/EOS")
    source, target = tokenizer.encode(article), tokenizer.encode(headline)
    return {"source": source[:source_cap], "decoder_input": [BOS] + target[:target_cap - 1],
            "labels": target[:target_cap - 1] + [EOS]}, len(source) > source_cap, len(target) > target_cap - 1


def prepare(archive, output_dir):
    import sentencepiece as spm
    archive, output = Path(archive), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Prepared output directory must be fresh")
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive hash differs from pinned publisher benchmark")
    raw_splits, raw_hashes, raw_sizes = {}, {}, {}
    with zipfile.ZipFile(archive) as compressed:
        for split in ["train", "dev", "test"]:
            content = compressed.read(f"ml_{split}.jsonl")
            raw_hashes[f"ml_{split}.jsonl"] = hashlib.sha256(content).hexdigest()
            raw_sizes[f"ml_{split}.jsonl"] = len(content)
            raw_splits[split] = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if {s: len(r) for s, r in raw_splits.items()} != {"train": 10358, "dev": 5388, "test": 5220}:
        raise ValueError("Publisher split counts changed")
    splits, records, audit = deduplicate(raw_splits)
    write_json(output / "data_audit.json", audit)
    write_jsonl(output / "record_manifest.jsonl", records)
    for split, rows in splits.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    corpus = output / "tokenizer-training.txt"
    with corpus.open("w", encoding="utf-8", newline="\n") as stream:
        for row in splits["train"]:
            stream.write(row["article"] + "\n" + row["headline"] + "\n")
    tokenizer_settings = dict(vocab_size=8000, model_type="bpe", character_coverage=1.0,
                              byte_fallback=True, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
                              normalization_rule_name="identity", remove_extra_whitespaces=False,
                              shuffle_input_sentence=False, input_sentence_size=0, num_threads=1,
                              max_sentence_length=1000000, hard_vocab_limit=False)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(output / "tokenizer"), **tokenizer_settings)
    tokenizer = Tokenizer(output / "tokenizer.model")
    truncation = {}
    for split, rows in splits.items():
        cached, source_truncated, target_truncated = [], 0, 0
        for row in rows:
            encoded, source_cut, target_cut = encode_example(row["article"], row["headline"], tokenizer)
            source_truncated += source_cut
            target_truncated += target_cut
            cached.append({"identity": row["identity"], **encoded})
        write_jsonl(output / f"{split}_tokens.jsonl", cached)
        truncation[split] = {"rows": len(rows), "source_truncated": source_truncated,
                             "target_truncated": target_truncated,
                             "source_truncation_rate": source_truncated / len(rows),
                             "target_truncation_rate": target_truncated / len(rows)}
    manifest = {
        "dataset": "AI4Bharat IndicHeadlineGeneration Malayalam v1.0, deduplicated official-split variant",
        "dataset_revision": REVISION, "archive_sha256": file_hash(archive), "archive_bytes": archive.stat().st_size,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "archive_url": f"https://huggingface.co/datasets/ai4bharat/IndicHeadlineGeneration/resolve/{REVISION}/data/ml_IndicHeadlineGeneration_v1.0.zip",
        "license": "CC-BY-NC-4.0; noncommercial research; see publisher dataset card",
        "raw_member_sha256": raw_hashes, "raw_member_bytes": raw_sizes, "tokenizer_settings": tokenizer_settings,
        "sentencepiece_version": spm.__version__, "vocab_size": tokenizer.vocab_size,
        "source_cap": SOURCE_CAP, "target_cap": TARGET_CAP, "truncation": truncation,
        "lead_words": max(1, round(statistics.median(len(row["headline"].split()) for row in splits["train"]))),
        "files": {file.name: file_hash(file) for file in sorted(output.iterdir()) if file.is_file()},
    }
    write_json(output / "prepared_manifest.json", manifest)
    print(json.dumps({"retained_rows": {name: len(rows) for name, rows in splits.items()},
                      "lead_words": manifest["lead_words"], "vocab_size": tokenizer.vocab_size}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
