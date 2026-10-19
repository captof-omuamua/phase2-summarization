# Data

The supported rerun uses **AI4Bharat IndicHeadlineGeneration, Malayalam v1.0**,
[publisher dataset card](https://huggingface.co/datasets/ai4bharat/IndicHeadlineGeneration),
revision `d9845634dc0f9cb48d4a26c9f6d8986fb87d2027`. The publisher states
**CC-BY-NC-4.0**; retain attribution and noncommercial research use.

The pinned ZIP contains `ml_train.jsonl`, `ml_dev.jsonl`, and `ml_test.jsonl`, with
10,358, 5,388 and 5,220 rows. Raw fields are `id`, `Document`, `Title`, `URL`.
IDs are local to each split. The raw archive is not modified or committed.

Archive SHA-256:
`1c371ca6a152238aa6419cb334e70f58d36cd31a1a7a12ef59909ada243fcaa6`.

After NFC/whitespace normalization, transitive article-hash/URL duplicate groups
use test>dev>train precedence and lowest composite identity. The resulting
single-reference benchmark variant has **7,212 train, 1,176 dev, 1,247 test**
records. These scores are not directly comparable with unmodified official
splits. Full record exclusions and content/source hashes are in the local prepared
manifest; exact duplicates do not rule out near-duplicates or same-event overlap.

The 8,000-piece tokenizer sees retained training articles/headlines only.
Preparation keeps numbers, Latin names, punctuation and Malayalam marks. The
256-source/64-target subword caps and split truncation counts are recorded.

`dataset-manifest.json` is retained **only as historical metadata** for the
excluded 89,934-row `news_90k_clean.csv`. Its provenance and redistribution terms
were not established; it is not the dataset used by the new measured experiment.
Do not combine its counts or unidentified scores with the publisher benchmark.
