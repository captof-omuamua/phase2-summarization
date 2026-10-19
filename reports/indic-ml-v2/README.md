# Malayalam headline generation: measured rerun

The corrected custom Transformer was trained from scratch on the retained real
Malayalam publisher benchmark and evaluated after checkpoint/decoding selection
was frozen. **It underperformed the simple Lead-8 baseline on every reported
overlap metric.** The implementation and evaluation repairs do not establish a
performance improvement or deployment-quality system.

| System | Test examples | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Corpus chrF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom Transformer, greedy | 1,247 | 7.738438 | 2.353682 | 7.600939 | 15.567714 |
| First 8 article words | 1,247 | **22.068678** | **10.203374** | **20.998934** | **34.695107** |

All overlap metrics use a 0–100 scale. ROUGE is macro F1 over case-sensitive
Unicode letter/mark/number runs after NFC normalization, with no English
stemming. Corpus chrF uses order 6, word order 0, beta 2 and no whitespace; its
recorded signature is
`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.

The separate test teacher-forced token cross-entropy was **5.853227759364205**
(24,026 supervised target tokens). Empty prediction rate was 0%; **85.5654%**
terminated with EOS. Generated token lengths had minimum 4, median 15, mean
22.8115 and maximum 64. These diagnostic values do not establish fluency,
factual correctness or usefulness.

## Data and fixed method

Source: [AI4Bharat IndicHeadlineGeneration](https://huggingface.co/datasets/ai4bharat/IndicHeadlineGeneration),
Malayalam v1.0, pinned revision
`d9845634dc0f9cb48d4a26c9f6d8986fb87d2027`. The archive SHA-256 is
`1c371ca6a152238aa6419cb334e70f58d36cd31a1a7a12ef59909ada243fcaa6`.
The publisher states **CC-BY-NC-4.0**; retain attribution and noncommercial use.

The official raw train/dev/test counts were 10,358/5,388/5,220. Conservative
NFC/whitespace normalization and transitive equal-article-or-canonical-URL
components produced **7,212/1,176/1,247** retained rows. Test>dev>train priority,
then lowest composite `(split,id)`, chose one representative per component before
fitting. Some duplicate groups have differing titles; references were not chosen
by generation quality. This **deduplicated single-reference official-split
variant is not directly comparable to unmodified benchmark scores**, and it is
separate from the historical unidentified 89,934-row corpus.

The 8,000-piece SentencePiece BPE saw retained training articles/headlines only.
PAD/UNK/BOS/EOS were 0/1/2/3. The custom model used width 256, four heads, three
independent encoder and decoder layers, feed-forward width 512, dropout 0.1 and
**8,057,664 parameters**. AdamW learning rate 3e-4, weight decay 0.01, clipping
1.0, batch 32, seed 42, validation-only scheduling/early stopping and a fixed
maximum of 20 epochs were declared before the measured run. Greedy decoding used
at most 64 generated tokens and each example's first EOS. The input-only baseline
length, eight words, came from the rounded median training headline length.

Checkpoint selection used validation **token-weighted cross-entropy only**. The
run stopped after 19 epochs and selected **epoch 16**, validation cross-entropy
**5.92214983215777**. The selection record was saved at
2026-10-19 03:38:57 UTC; final test evaluation started at 03:39:15 UTC. No test
score influenced vocabulary, architecture, training settings or decoder choices.

## Runtime and reproducibility

Measured training took **143.614 seconds**. Test generation took **12.214 seconds**
and complete evaluation **13.716 seconds**. Environment: Python 3.12.14,
PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, NVIDIA GeForce RTX 5090 Laptop GPU,
bfloat16. Strict deterministic algorithms, deterministic cuDNN and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` were recorded. Source snapshots, tokenizer,
prepared splits, protocol, selection and predictions are checksum-bound.

An initial strict training run was superseded **before any test evaluation**
solely to preserve the hashed `.gitattributes` file's LF bytes on Windows
checkouts. A fresh run used identical modeling settings. All 19 epoch losses,
token counts, learning rates and all 94 selected state-dictionary tensors matched
exactly between these two runs in this environment. This is observed local
repeatability, not a guarantee across hardware or library versions.

Independent checks reconstructed the raw duplicate components and retained
identities, verified the training-only tokenizer corpus/caches, and recomputed
all metrics from the saved predictions using a separate Counter/LCS/character
ngram implementation. References, source identities, complete 1,247-row coverage
and the input-only baseline were also verified. A fresh process reproduced the
first complete 32-example batch's strings, lengths and EOS flags exactly from
the selected checkpoint and frozen precision. Eight correctness tests additionally
exercise layer independence/gradients, masks, EOS supervision/termination,
target-independent grouping, Unicode metrics, safe imports and a tiny actual-model
overfit/checkpoint round trip.

## Limits and evidence

The source cap truncated **77.94% train, 79.08% dev and 77.87% test** articles.
Headline labels were truncated for three training examples and no dev/test
examples. These are truncated-input headline results, not full-document summaries.
Exact deduplication does not establish event-level independence; there may be
near-duplicate stories or source biases. One seed, one benchmark variant and
automatic overlaps provide limited evidence. No fluent-language human assessment
or factuality evaluation was established. The broader multimodal proposal remains
unimplemented in this text-only experiment.

Full-precision and provenance evidence:

- [Data audit](data_audit.json), [prepared dataset/tokenizer manifest](prepared_manifest.json),
  [independent preparation checks](prepared_independent_review.json).
- [Frozen protocol](protocol.json), [epoch history](history.json),
  [selection](selection.json), [training status](training_status.json),
  [captured training output](training_log.txt), [installed versions](environment.txt).
- [Final metrics](metrics.json), [saved-file metric recomputation](recomputed_metrics.json),
  [evaluation status](evaluation_status.json), [independent metric/coverage checks](independent_results.json).
- [Repeatability checks](repeatability_review.json),
  [fresh-process checkpoint check](checkpoint_reload_review.json),
  [deterministic examples](deterministic_examples.jsonl).
- [Artifact hashes and sizes](artifact_manifest.json), including original local
  artifact checksums and LF-normalized repository text-copy checksums.

The repository contains small aggregates and fixed examples only. Full prepared
data/token caches, tokenizer, checkpoint and test predictions remain local and in
the downloadable evidence bundle; they are not missing from the original run.
Reproduction commands are in the main README. The five examples were fixed by
stable identity order before scoring, rather than selected as successes, and
remain subject to the dataset's attribution/noncommercial terms.
