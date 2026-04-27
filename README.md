# NewsBot Intelligence System 2.0
**ITAI 2373 — Final Project**  
Advanced NLP Integration and Analysis Platform

---

## Overview

NewsBot 2.0 is a production-ready news analysis platform that combines classical NLP techniques with local large language model inference. It classifies, summarizes, clusters, and converses about news articles — entirely offline, with no external API dependencies.

Built on top of the NewsBot 1.0 midterm foundation, this system adds four integrated modules:

| Module | Capability |
|--------|-----------|
| A — Content Analysis | Topic modeling (LDA/NMF), sentiment evolution, entity mapping |
| B — Language Models | Summarization, insight generation, semantic search via local Llama |
| C — Multilingual | Language detection, translation, cross-lingual analysis |
| D — Conversational Interface | Natural language queries, intent parsing, interactive exploration |

> **Local-first design:** Module B uses [ollama](https://ollama.com/) to run Llama 3.2 locally on CPU. No OpenAI key required.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ITAI2373-NewsBot-Final.git
cd ITAI2373-NewsBot-Final
pip install -r requirements.txt
```

### 2. Set up ollama (for Module B)

```bash
# Install ollama: https://ollama.com/download
ollama pull llama3.2        # ~2GB, runs on CPU
ollama serve                # keep running in a separate terminal
```

### 3. Configure API keys (optional)

```bash
cp config/api_keys_template.txt config/api_keys.txt
# Edit api_keys.txt with any optional keys (e.g. translation APIs)
```

### 4. Run the notebooks in order

```
notebooks/01_Data_Exploration.ipynb       ← start here
notebooks/02_Advanced_Classification.ipynb
notebooks/03_Topic_Modeling.ipynb
notebooks/04_Language_Models.ipynb        ← Module B demo
notebooks/05_Multilingual_Analysis.ipynb
notebooks/06_Conversational_Interface.ipynb
notebooks/07_System_Integration.ipynb     ← full pipeline
```

---

## Project Structure

```
ITAI2373-NewsBot-Final/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.py              # Centralized configuration
│   └── api_keys_template.txt    # Template — never commit real keys
├── src/
│   ├── data_processing/
│   │   ├── text_preprocessor.py
│   │   ├── feature_extractor.py
│   │   └── data_validator.py
│   ├── analysis/
│   │   ├── classifier.py
│   │   ├── sentiment_analyzer.py
│   │   ├── ner_extractor.py
│   │   └── topic_modeler.py
│   ├── language_models/         ← Module B
│   │   ├── summarizer.py        # NewsSummarizer
│   │   ├── generator.py         # ContentGenerator
│   │   └── embeddings.py        # ArticleEmbedder
│   ├── multilingual/
│   │   ├── translator.py
│   │   ├── language_detector.py
│   │   └── cross_lingual_analyzer.py
│   ├── conversation/
│   │   ├── query_processor.py
│   │   ├── intent_classifier.py
│   │   └── response_generator.py
│   └── utils/
│       ├── visualization.py
│       ├── evaluation.py
│       └── export.py
├── notebooks/                   # 7 annotated Jupyter notebooks
├── tests/                       # Unit tests
├── data/
│   ├── raw/                     # Original BBC News dataset
│   ├── processed/               # Cleaned data
│   ├── models/                  # Saved model files
│   └── results/                 # Analysis outputs
├── docs/                        # Technical and user documentation
└── reports/                     # Executive summary, technical report
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |
| GPU | Not required | Optional (speeds up Module B) |

**CPU performance estimates (Module B, llama3.2:3b):**
- Summarization: ~8–15 sec/article
- Insight generation (10 articles): ~30–60 sec
- Embeddings (sentence-transformers): <1 sec/article

---

## Dataset

This project uses the **BBC News Dataset** (BBC, 2004–2005), containing 2,225 articles across 5 categories: business, entertainment, politics, sport, tech.

Place the raw data at `data/raw/bbc/` before running notebooks.

---

## Module B: Local LLM Design Decision

Module B uses two different local models intentionally:

**Llama 3.2 (via ollama)** handles generative tasks — summarization, content enhancement, insight generation, and query understanding — where language fluency matters.

**sentence-transformers** (`all-MiniLM-L6-v2`) handles embeddings and semantic search. Dedicated embedding models produce significantly better similarity scores than using a generative model's hidden states, and run ~10x faster on CPU.

---

## Individual Contributions

See `docs/individual_contributions.md` for a breakdown of contributions per team member.

---

## Academic Integrity

All core NLP components are original implementations. External libraries are documented in `requirements.txt`. AI assistance (GitHub Copilot, Claude) was used for boilerplate generation and is disclosed per HCC policy.

---

## License

For academic use only. Dataset © BBC.