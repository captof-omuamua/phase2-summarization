"""Single-article inference using the same frozen model, tokenizer and greedy decoder."""

import argparse
import json
from pathlib import Path

import torch

from preprocess import Tokenizer, normalize_text
from train import load_model, precision_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--article-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _, manifest = load_model(args.checkpoint, args.prepared_dir, device)
    tokenizer = Tokenizer(args.prepared_dir / "tokenizer.model")
    article = normalize_text(args.article_file.read_text(encoding="utf-8"))
    if not article:
        raise ValueError("Article must not be empty")
    source = tokenizer.encode(article)[:manifest["source_cap"]]
    protocol = json.loads((args.checkpoint.parent / "protocol.json").read_text(encoding="utf-8"))
    precision = protocol["environment"]["precision"]
    if precision == "bfloat16" and device.type != "cuda":
        raise ValueError("Frozen bfloat16 inference requires CUDA")
    with precision_context(device, precision):
        ids = model.generate_ids(torch.tensor([source], dtype=torch.long, device=device), max_length=64)
    print(tokenizer.decode(ids[0, 1:].tolist()))


if __name__ == "__main__":
    main()
