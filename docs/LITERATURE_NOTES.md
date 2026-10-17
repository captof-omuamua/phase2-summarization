# Literature reading notes

Recovered notes about external research papers. These are literature summaries, not results produced by this project. Numerical claims below belong to the summarized sources and have not been independently verified during this archive preparation.

The first paper is **Advanced multiple document summarization via iterative recursive transformer networks and multimodal transformer**, DOI: [10.7717/peerj-cs.2463](https://doi.org/10.7717/peerj-cs.2463). The original reference PDF is not redistributed here.

---

Advanced Multiple Document Summarization via Iterative Recursive Transformer Networks and Multimodal Transformer
Abstract
The paper addresses the challenges of summarizing vast and diverse digital information by introducing innovative methodologies. These include Recursive Transformer Networks (ReTran) and multimodal transformers integrating text, images, and metadata. The model leverages actor-critic reinforcement learning, zero-shot summarization, and knowledge-enhanced transformers, offering notable improvements in ROUGE scores and semantic coherence.
Key Contributions
Recursive Transformer Networks (ReTran):
Combines recursive neural networks and transformer architectures.
Enhances understanding of local and global textual dependencies.
Improves ROUGE scores by 5–10%.
Multimodal Transformers with Cross-Modal Attention:
Integrates text, images, and metadata for holistic summaries.
Improves quality metrics by 8–12%.
Actor-Critic Reinforcement Learning:
Optimizes summary quality through iterative feedback.
Outperforms Q-learning-based strategies by 5–8%.
Meta-Learning for Zero-Shot Summarization:
Enables summarization in unseen domains with minimal data.
Provides a performance uptick of 6–10%.
Knowledge-Enhanced Transformers:
Incorporates external knowledge for semantic depth.
Boosts ROUGE scores by 7–12%.

Datasets Used
CNN/Daily Mail: News articles and highlights.
Multi30k: Image-text pairs for multimodal evaluation.
XSum: Extreme summarization dataset with BBC articles.
RDF2Text: Tests knowledge integration via RDF triples.
Custom Domain-Specific Datasets: Includes medical, legal, and scientific texts for zero-shot evaluation.

Evaluation Metrics
ROUGE Scores: ROUGE-1, ROUGE-2, ROUGE-L.
BLEU Scores: Linguistic quality of generated text.
Human Evaluation: Assessing coherence, relevance, and informativeness.

Results
The proposed model outperformed benchmarks across all datasets. For instance:
CNN/Daily Mail: Achieved a ROUGE-1 score of 42.9 (vs. 40.6 from the best benchmark).
Multi30k: Achieved a BLEU score of 62.3 (vs. 58.7 from the best benchmark).

Significance and Applications
This work represents a significant leap in multi-document summarization, combining multimodal integration, zero-shot adaptability, and external knowledge use. Applications include news summarization, technical document abstraction, and domain-specific summarization tasks.

 Text-Image-Video Summary Generation Using Joint Integer Linear Programming (JILP)
Abstract
The paper introduces a novel approach for generating summaries from multi-modal data (text, images, videos). It proposes a new task of Text-Image-Video Summary Generation (TIVS) and presents a Joint Integer Linear Programming (JILP) framework to address it. The dataset developed includes text documents, images, and videos, and the model is evaluated based on salience, diversity, and cross-modal correspondence.

Key Contributions
Problem Definition:
Introduced a novel task of generating multi-modal summaries.
Defined objectives: salience, diversity, and correspondence across modalities.
Framework:
Joint Integer Linear Programming (JILP) for extracting salient, diverse, and correlated data.
Preprocessing includes ASR (audio-to-text conversion) and key-frame extraction for videos.
Dataset:
Extended an existing multi-modal dataset to include summaries with text, images, and videos.
Annotated by three undergraduate students for relevance scoring.
Performance Metrics:
Evaluated using ROUGE scores for text and precision-recall for images.
Achieved higher scores in experiments compared to baseline methods.
Results
Text Summary: Outperformed baselines in ROUGE-2 and ROUGE-L metrics.
Image Summary: Precision and recall improved significantly with optimized parameters.
Video Selection: 44% accuracy in selecting the most relevant video compared to a random baseline (16%).

Conclusion
This work pioneers a fully multi-modal summarization system capable of integrating text, images, and videos. It demonstrates superior performance through the proposed JILP framework, setting a foundation for future research in multi-modal summarization.




