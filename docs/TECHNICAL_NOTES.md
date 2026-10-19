# Technical notes

The supported pipeline has been rebuilt around the existing custom Transformer.
Legacy checkpoints used shared layer objects and PAD=EOS and are incompatible
with the independent-layer, distinct-special-token configuration.

Encoder/decoder stacks deep-copy their layer templates. All three layers have
independent parameter storage and receive gradients. Attention capture is opt-in;
the normal training path does not retain attention autograd graphs. Captured
MultiheadAttention weights use PyTorch's batch/target/source dimensions, with
averaged heads, rather than the old visualization's assumed axes.

Sources and decoder inputs use PAD=0, while labels use -100 only at padded
positions. EOS=3 is a supervised target, BOS=2 begins decoder inputs, and UNK=1
is distinct. Boolean causal masks prohibit future-target attention; source/memory
padding masks prevent padded positions affecting unpadded outputs. Loss reporting
sums token losses and divides by nonpadding target-token count, rather than
averaging unequal batch means.

The official benchmark splits remain separate after deterministic transitive
article/URL duplicate-component cleanup. IDs are `(split,id)`. Held-out test
reference titles are used only for integrity diagnostics and final scoring;
they do not determine grouping, tokenizer training, training inputs, generation
or checkpoint selection. Training titles provide supervised targets and tokenizer
text; dev titles supply validation loss. Within a duplicate component,
test>dev>train precedence and lowest composite identity choose a single record;
differing reference titles are audited, not selected by output quality.

Validation loss selects the checkpoint. Test generation occurs only after the
selection record and decoding settings are frozen. Greedy decoding tracks an
accumulated finished mask, pads finished examples and trims at each first EOS.
The old eight-way decoding comparison and automatic multilingual BERTScore
downloads are retired from the supported pipeline.

ROUGE uses case-sensitive NFC Unicode letter/mark/number runs and no English
stemmer. chrF is corpus-level character order 6, beta 2, word order 0, whitespace
disabled. Empty predictions count in the denominator and score zero overlap.
Generation failures stop evaluation; error strings are never scored as summaries.

Preparation artifacts, tokenizer, source snapshots, protocol and selected model
are checksum-bound. Loading/evaluation rejects mismatched artifacts or changed
source code. The last checkpoint records optimizer/scheduler and random/data-loader
states for explicit resume; the best checkpoint remains selected by validation
loss. No computation or network/file access is triggered merely by importing the
supported modules.

Exact duplicate cleanup cannot establish event-level independence. The 256-token
source cap truncates many articles; overlap metrics do not establish factual
faithfulness or naturalness. There is no fluent-language human evaluation or
claim that the new benchmark reproduces the unknown-provenance historical corpus.
