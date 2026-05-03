# Individual Contributions

## Team Member: Jemima Egwurube

### Modules Owned
| Module | Description |
|--------|-------------|
| Module A | Topic Modeling — LDA, NMF, clustering, topic evolution |
| Module B | Language Understanding — summarization, Q&A, insights, web frontend |
| System Integration | End-to-end pipeline, df_final, notebook structure |

### Key Contributions
- Designed and implemented `TopicModeler` class (LDA + NMF unified API)
- Built topic evolution visualization pipeline (heatmaps, stacked charts)
- Implemented K-Means content clustering with silhouette-based k selection
- Developed Module B LLM pipeline using ollama/llama3.2
- Built `ArticleQueryEngine` for stateful multi-turn article Q&A
- Created production web frontend (`NewsBot_IntelligenceSystem_2.html`)
- Authored technical documentation and API reference

### Hours Invested
*[Update with your actual hours]*

### Challenges Overcome
- Integrating LDA's probabilistic output with NMF's deterministic output in a
  unified TopicModeler API while keeping both models' evaluation metrics accessible
- Handling ollama JSON parse failures from llama3.2 — implemented graceful
  fallback to raw text extraction
- Designing the web app's multi-turn Q&A to maintain conversation state across
  Claude API calls without server-side storage

### What I Learned
- Practical differences between LDA and NMF in terms of topic interpretability
  and hyperparameter sensitivity on news corpora
- How to prompt LLMs for structured JSON output and handle parse failures robustly
- The importance of grounding LLM analysis in pre-computed quantitative signals
  (sentiment scores, TF-IDF terms) to reduce hallucination
