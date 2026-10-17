# Phase 2 proposal

The January 4, 2025 zeroth-review presentation was titled Multi-Document Summarization Using Recursive Transformer Networks and Multimodal Transformer.

## Proposed scope

- Low-resource Dravidian-language summarization: Tamil, Telugu, and Malayalam.
- Combining text, images, and metadata.
- Recursive representations and cross-modal attention.
- Graph-based extractive summaries and BART-based abstractive summaries.
- Knowledge integration, zero-shot adaptation, and feedback-based refinement.

These are research objectives described by the slides. They are not all implemented by the available Python scripts. The recovered implementation is text-only Malayalam paragraph-to-headline generation with a custom Transformer.

## Evidence mapping

- Five root Python scripts: implemented text-processing, model, training, and decoding logic.
- January review diagrams: original proposed architecture, extracted from the clean presentation version. Team names, student IDs, group identifiers, and footers were excluded.
- Turkish notebook: preliminary dataset loading only; no completed training pipeline.
- Literature notes: summaries of external publications, not project experiments.
- Dataset manifest: metadata calculated from the locally available cleaned CSV.

The two review PDFs are retained at their source, including an annotated copy. Neither original PDF is uploaded because it contains personal identifiers. No standalone summarization results were recovered.
