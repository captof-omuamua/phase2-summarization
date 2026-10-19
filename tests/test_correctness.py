import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from model import TransformerSummarizer
from preprocess import BOS, EOS, PAD, UNK, canonical_url, deduplicate, encode_example, normalize_text
from train import collate_examples, forward_loss
from evaluate import UnicodeTokenizer, score_predictions


def tiny_model():
    return TransformerSummarizer(vocab_size=16, d_model=8, nhead=2, num_encoder_layers=3,
                                 num_decoder_layers=3, dim_feedforward=16, dropout=0)


def test_independent_layers_and_gradients():
    torch.manual_seed(42)
    model = tiny_model()
    for layers in [model.encoder.layers, model.decoder.layers]:
        assert len({id(layer) for layer in layers}) == 3
        assert len({layer.linear1.weight.data_ptr() for layer in layers}) == 3
        original = layers[1].linear1.weight.detach().clone()
        with torch.no_grad():
            layers[0].linear1.weight.add_(0.1)
        torch.testing.assert_close(layers[1].linear1.weight, original)
    batch = collate_examples([{"source": [4, 5, 6], "decoder_input": [BOS, 8], "labels": [8, EOS]},
                              {"source": [6], "decoder_input": [BOS], "labels": [EOS]}])
    total, tokens = forward_loss(model, batch)
    (total / tokens).backward()
    for layer in list(model.encoder.layers) + list(model.decoder.layers):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        assert layer.linear1.weight.grad.abs().sum() > 0
    assert model.attention_weights == {"encoder": [], "decoder_self": [], "decoder_cross": []}


def test_special_ids_shift_eos_supervision_and_only_padding_ignored():
    assert len({PAD, UNK, BOS, EOS}) == 4
    fake = types.SimpleNamespace(encode=lambda text: {"article": [4, 5, 6, 7], "title": [8, 9, 10]}[text])
    row, source_cut, target_cut = encode_example("article", "title", fake, source_cap=3, target_cap=3)
    assert row == {"source": [4, 5, 6], "decoder_input": [BOS, 8, 9], "labels": [8, 9, EOS]}
    assert source_cut and target_cut
    batch = collate_examples([row, {"source": [4], "decoder_input": [BOS], "labels": [EOS]}])
    assert batch["labels"].tolist() == [[8, 9, EOS], [EOS, -100, -100]]
    assert int(batch["labels"].eq(EOS).sum()) == 2
    assert int(batch["labels"].ne(-100).sum()) == 4
    with pytest.raises(ValueError, match="distinct"):
        TransformerSummarizer(16, eos_id=0, pad_id=0)


def test_source_padding_invariance_causal_mask_and_token_weighted_loss():
    torch.manual_seed(3)
    model = tiny_model().eval()
    source = torch.tensor([[4, 5, 6]])
    padded = torch.tensor([[4, 5, 6, PAD, PAD]])
    target = torch.tensor([[BOS, 7]])
    causal = model._generate_square_subsequent_mask(2)
    assert causal.dtype == torch.bool and causal.tolist() == [[False, True], [False, False]]
    def logits(src, tgt):
        return model(src, tgt, src_key_padding_mask=src.eq(PAD), memory_key_padding_mask=src.eq(PAD),
                     tgt_mask=model._generate_square_subsequent_mask(tgt.size(1)))
    with torch.no_grad():
        torch.testing.assert_close(logits(source, target), logits(padded, target), atol=1e-6, rtol=1e-5)
        changed = target.clone()
        changed[:, 1] = 9
        torch.testing.assert_close(logits(source, target)[:, 0], logits(source, changed)[:, 0])
    rows = [{"source": [4, 5], "decoder_input": [BOS, 8, 9], "labels": [8, 9, EOS]},
            {"source": [6], "decoder_input": [BOS], "labels": [EOS]}]
    loss, count = forward_loss(model, collate_examples(rows))
    separate = [forward_loss(model, collate_examples([row])) for row in rows]
    assert count == 4
    assert float(loss.detach()) / count == pytest.approx(sum(float(x.detach()) for x, _ in separate) / sum(n for _, n in separate), abs=1e-6)


def test_greedy_freezes_each_sequence_after_first_eos_and_caps_length():
    model = tiny_model()
    model.output_proj = torch.nn.Identity()
    def encode(self, src, **kwargs):
        return torch.zeros(1, src.size(0), 16), []
    def decode(self, target, memory, **kwargs):
        step, batch = target.size(1), target.size(0)
        output = torch.full((step, batch, 16), -100.0)
        output[-1, 0, EOS if step == 1 else 7] = 100
        output[-1, 1, EOS if step == 3 else 8] = 100
        return output, [], []
    model.encode = types.MethodType(encode, model)
    model.decode = types.MethodType(decode, model)
    result = model.generate_ids(torch.tensor([[4], [5]]), max_length=5)
    assert result.tolist() == [[BOS, EOS, PAD, PAD], [BOS, 8, 8, EOS]]
    capped = model.generate_ids(torch.tensor([[4], [5]]), max_length=2)
    assert capped.tolist() == [[BOS, EOS, PAD], [BOS, 8, 8]]


def sample_splits():
    row = lambda i, text, url, title="തലക്കെട്ട്": {"id": str(i), "Document": text, "Title": title, "URL": url}
    return {
        "test": [row(1, "പാഠം ഒന്ന്", "https://example.org/a"), row(2, "പാഠം രണ്ട്", "https://example.org/b")],
        # Bridge: dev:1 connects test:1 via URL; train:1 connects dev:1 via article.
        "dev": [row(1, "ബ്രിഡ്ജ് പാഠം", "https://example.org/a#x"), row(2, "വേറെ വാർത്ത", "https://example.org/c")],
        "train": [row(1, "ബ്രിഡ്ജ്   പാഠം", "https://example.org/d"), row(2, "പരിശീലന വാർത്ത", "https://example.org/e")],
    }


def test_transitive_dedup_precedence_local_ids_and_target_independence():
    raw = sample_splits()
    kept, manifest, audit = deduplicate(raw)
    assert {s: [r["identity"] for r in rows] for s, rows in kept.items()} == {
        "test": ["test:1", "test:2"], "dev": ["dev:2"], "train": ["train:2"]}
    assert {r["identity"]: r["representative"] for r in manifest}["train:1"] == "test:1"
    assert audit["retained_article_and_nonempty_url_overlap"] == 0
    raw["test"][0]["Title"] = "മാറ്റിയ ശീർഷകം"
    altered, changed, _ = deduplicate(raw)
    assert kept["train"] == altered["train"]
    assert [(r["identity"], r["retained"]) for r in manifest] == [(r["identity"], r["retained"]) for r in changed]
    assert canonical_url("HTTPS://Example.org/a?utm_source=x&story=12#part") == "https://example.org/a?story=12"
    assert "2026 ABC" in normalize_text("വാർത്ത  2026 ABC!")


def test_unicode_rouge_chrf_exact_unrelated_and_empty_predictions():
    text = "മലയാളം വാർത്ത ഇന്ന്"
    assert UnicodeTokenizer().tokenize(text) == text.split()
    records = [{"identity": "test:1", "reference": text, "transformer": text, "lead_baseline": ""}]
    score, _ = score_predictions(records)
    for metric in ["rouge1", "rouge2", "rougeL", "chrF"]:
        assert score["systems"]["transformer"][metric] == pytest.approx(100)
        assert score["systems"]["lead_baseline"][metric] == 0
    assert score["systems"]["lead_baseline"]["empty_prediction_rate"] == 1
    records[0]["transformer"] = "മറ്റൊരു കഥ മാത്രം"
    other, _ = score_predictions(records)
    assert other["systems"]["transformer"]["rouge1"] < 100


def test_checkpoint_roundtrip_and_tiny_actual_model_overfit(tmp_path):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = tiny_model()
    batch = collate_examples([{"source": [4, 5], "decoder_input": [BOS, 8], "labels": [8, EOS]}])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    first = float((forward_loss(model, batch)[0] / 2).detach())
    for _ in range(35):
        optimizer.zero_grad()
        total, count = forward_loss(model, batch)
        (total / count).backward()
        optimizer.step()
    model.eval()
    last = float((forward_loss(model, batch)[0] / 2).detach())
    assert last < first * 0.25
    torch.save(model.state_dict(), tmp_path / "model.pt")
    reloaded = tiny_model().eval()
    reloaded.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    assert model.generate_ids(batch["source"], max_length=6).tolist() == reloaded.generate_ids(batch["source"], max_length=6).tolist()


def test_imports_are_safe_without_inputs_or_network(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = '''
import socket,torch,sentencepiece
def forbidden(*args,**kwargs):
    raise AssertionError("Import attempted work")
socket.create_connection=forbidden
torch.load=forbidden
torch.save=forbidden
sentencepiece.SentencePieceTrainer.train=forbidden
import model,preprocess,train,evaluate,test
from pathlib import Path
assert not list(Path('.').iterdir())
'''
    environment = {**os.environ, "PYTHONPATH": str(root)}
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, check=True, timeout=60)
