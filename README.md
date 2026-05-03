# NewsBot Intelligence System 2.0
### ITAI 2373 — Final Project | Houston Community College

> A production-ready news analysis platform demonstrating advanced NLP techniques including
> topic modeling, language model integration, multilingual analysis, and conversational AI.

---

## Project Overview

NewsBot 2.0 extends the midterm pipeline into a full-stack NLP intelligence system built on
the BBC News dataset (5 categories: tech, business, politics, sport, entertainment).

The system integrates four major modules:

| Module | Capability | Key Techniques |
|--------|-----------|----------------|
| **A** | Advanced Content Analysis | LDA, NMF, K-Means clustering |
| **B** | Language Understanding & Generation | LLM summarization, Q&A, insights |
| **C** | Multilingual Intelligence | Language detection, translation |
| **D** | Conversational Interface | Intent classification, context management |

---

## System Architecture

```
NewsBot 2.0
│
├── Data Layer          BBC News Dataset (Kaggle) → raw/ → processed/
├── Analysis Engine     Preprocessing → TF-IDF → POS → Sentiment → NER → Topics
├── LLM Layer           ollama (llama3.2) for summarization, Q&A, insight generation
├── Multilingual Layer  langdetect + deep-translator for cross-language analysis
├── Conversation Layer  Intent classification → context-aware response generation
└── Interface           Web app (HTML/JS/React) + Jupyter notebooks
```

---

## Repository Structure

```
ITAI2373-NewsBot-Final/
├── README.md                        # This file
├── requirements.txt                 # All dependencies with versions
├── config/
│   ├── settings.py                  # Central configuration
│   └── api_keys_template.txt        # API key setup guide (no real keys)
├── src/
│   ├── data_processing/
│   │   ├── text_preprocessor.py     # Enhanced from midterm
│   │   ├── feature_extractor.py     # TF-IDF, embeddings, custom features
│   │   └── data_validator.py        # Data quality checks
│   ├── analysis/
│   │   ├── classifier.py            # Multi-class news classifier
│   │   ├── sentiment_analyzer.py    # VADER + enhanced sentiment
│   │   ├── ner_extractor.py         # Named entity recognition
│   │   └── topic_modeler.py         # LDA + NMF implementation
│   ├── language_models/
│   │   ├── summarizer.py            # Abstractive summarization
│   │   ├── generator.py             # Content enhancement & generation
│   │   └── embeddings.py            # Semantic similarity
│   ├── multilingual/
│   │   ├── translator.py            # Translation workflows
│   │   ├── language_detector.py     # Language identification
│   │   └── cross_lingual_analyzer.py
│   ├── conversation/
│   │   ├── query_processor.py       # NL query handling
│   │   ├── intent_classifier.py     # Intent detection
│   │   └── response_generator.py    # Response generation
│   └── utils/
│       ├── visualization.py         # Plotting utilities
│       ├── evaluation.py            # Model evaluation
│       └── export.py                # Report generation
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Advanced_Classification.ipynb
│   ├── 03_Topic_Modeling.ipynb      # Module A
│   ├── 04_Language_Models.ipynb     # Module B
│   ├── 05_Multilingual_Analysis.ipynb
│   ├── 06_Conversational_Interface.ipynb
│   └── 07_System_Integration.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   ├── test_topic_modeling.py
│   └── test_integration.py
├── data/
│   ├── raw/           # Original BBC dataset (not committed — see Setup)
│   ├── processed/     # Cleaned DataFrames
│   ├── models/        # Serialized model files
│   └── results/       # Analysis outputs and visualizations
└── docs/
    ├── technical_documentation.md
    ├── user_guide.md
    ├── api_reference.md
    └── deployment_guide.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Student-Portfolio-Repository.git
cd Student-Portfolio-Repository/ITAI2373-NewsBot-Final
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set up ollama (for Module B LLM features)

```bash
# Install ollama: https://ollama.ai
ollama pull llama3.2
ollama serve   # runs on localhost:11434
```

### 4. Download the BBC dataset

```bash
# Option A: Kaggle CLI
kaggle datasets download -d shivamkushwaha/bbc-full-text-document-classification --unzip -p data/raw/

# Option B: Manual download from https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification
# Place the CSV in data/raw/
```

### 5. Run the notebooks in order

Open in Google Colab or Jupyter Lab and run notebooks `01` through `07` sequentially.
Each notebook picks up `df_final` from the previous one.

---

## Module Descriptions

### Module A — Topic Modeling (`03_Topic_Modeling.ipynb`)
- LDA (Latent Dirichlet Allocation) with perplexity evaluation
- NMF (Non-negative Matrix Factorization) with reconstruction error
- Topic evolution heatmaps across BBC categories
- K-Means content clustering with silhouette analysis
- Interactive pyLDAvis visualization

### Module B — Language Models (`04_Language_Models.ipynb`)
- Abstractive summarization via llama3.2/ollama
- Contextual content enhancement (background, trends, implications)
- Multi-turn article Q&A with `ArticleQueryEngine`
- Structured insight generation with recommended queries
- Web application frontend (see `NewsBot_IntelligenceSystem_2.html`)

### Module C — Multilingual (`05_Multilingual_Analysis.ipynb`)
- Language detection with `langdetect`
- Translation via `deep-translator` (Google Translate API wrapper)
- Cross-lingual sentiment and entity comparison
- Multi-language topic distribution analysis

### Module D — Conversational Interface (`06_Conversational_Interface.ipynb`)
- Rule + embedding hybrid intent classifier
- Context-aware response generation
- Query expansion using topic model vocabulary
- Session history management

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | BBC News, 2,225 articles, 5 categories |
| LDA Perplexity | *Run notebook to generate* |
| NMF Reconstruction Error | *Run notebook to generate* |
| Clustering Silhouette Score | *Run notebook to generate* |
| Supported Languages (Module C) | 10+ via deep-translator |

---

## Dependencies

See `requirements.txt` for pinned versions. Core libraries:
`spacy`, `scikit-learn`, `nltk`, `vaderSentiment`, `ollama`,
`pyLDAvis`, `langdetect`, `deep-translator`, `matplotlib`, `seaborn`, `pandas`, `numpy`

---

## Individual Contributions

*[Update this section with your team's actual contribution breakdown]*

| Member | Modules | Key Contributions |
|--------|---------|-------------------|
| Jemima Egwurube | A, B, Integration | Topic modeling, LLM pipeline, web frontend |

See `docs/individual_contributions.md` for full breakdown.

---

## Academic Integrity

All core NLP implementations are original work. External libraries are attributed in
`requirements.txt` and inline code comments. AI assistance (Claude) was used for
scaffolding and debugging, documented per course policy.

---

## License

For academic use only — ITAI 2373, Houston Community College.
