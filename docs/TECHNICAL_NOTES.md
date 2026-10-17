# Technical notes

## Preserved implementation

The Python scripts are archived research code. The only intended code edit during organization was replacing the personal absolute input path in preprocess.py with news_90k.csv. The model and evaluation logic were retained. The unpinned dependency list is inferred from imports and is not an environment lockfile.

## Known reproduction issues

- Evaluation leakage: train.py shuffles the full dataset with random_state=42 and uses its first 90% for training. evaluate.py takes a 10% sample of the same full dataset with random_state=42. Those evaluation rows come from the beginning of that same permutation and overlap training data. Recover or define a genuinely held-out test set before reporting performance.
- Reused layers: model.py builds ModuleList by repeating the same encoder/decoder layer object, tying parameters across passes. This must be understood before interpreting layer counts or changing checkpoint compatibility.
- Language metrics: the default rouge_score tokenizer/stemmer is not configured for Malayalam script. Validate tokenization before trusting ROUGE values. The function named compute_indic_bleu uses ordinary sacreBLEU sentence scoring; it is not a separate Indic-language BLEU implementation.
- Attention visualization: MultiheadAttention uses its default averaged attention format, while plotting comments assume a different axis order. Verify the dimensions before interpreting plots.
- Runtime compatibility: scheduler options, tokenizer behavior, BERTScore language/model selection, and dependencies need verification in the chosen environment.
- Missing evidence: no matching checkpoint, trained-output examples, predictions CSV, or historical score table was recovered.
- Import side effects: train.py and evaluate.py perform substantial work at module scope. Do not import them merely to inspect functions.

## Intended generated artifacts

train.py writes best_model.pth, losses.csv, training_validation_loss.png, and attention_scores.png. Their presence in code does not mean these files were recovered. test.py and evaluate.py require a compatible checkpoint. Tokenizer access may involve external downloads.
