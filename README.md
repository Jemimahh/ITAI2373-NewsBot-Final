# NewsBot Intelligence System 2.0
### ITAI 2373 — Final Project | Natural Lanuage Processing

> A production-ready news analysis platform demonstrating advanced NLP techniques including
> topic modeling, language model integration, multilingual analysis, and conversational AI —
> with a Flask web application frontend and trained BBC News classifier.

---

## Live Demo

```bash
git clone https://github.com/Jemimahh/ITAI2373-NewsBot-Final.git

python -m venv newsbot_env && source newsbot_env/bin/activate   # Windows: newsbot_env\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
# open http://127.0.0.1:5000
```

> **LLM features** (summarize, chat) require ollama running locally: `ollama serve`
> Pull a model: `ollama pull llama3.2:1b`

---

## Project Overview

NewsBot 2.0 extends the ITAI 2373 midterm pipeline into a full-stack NLP intelligence
system trained on the BBC News dataset (2,225 articles, 5 categories).

| Module | Capability | Key Techniques |
|--------|-----------|----------------|
| **A** | Advanced Content Analysis | LDA, NMF, K-Means clustering, pyLDAvis |
| **B** | Language Understanding & Generation | LLM summarization, Q&A, insight generation |
| **C** | Multilingual Intelligence | Language detection, translation, cross-lingual sentiment |
| **D** | Conversational Interface | Intent classification, context management, corpus Q&A |
| **Bonus** | Flask Web Application | Full-stack app with trained classifier inference |

---

## Classifier Performance

Trained on BBC News dataset using TF-IDF features + Logistic Regression:

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.98%** |
| Model | Logistic Regression (multinomial) |
| Features | TF-IDF, max 5,000 terms, unigrams + bigrams |
| Training set | 1,780 articles |
| Test set | 445 articles |
| Categories | tech, business, politics, sport, entertainment |

---

## System Architecture

```
NewsBot Intelligence System 2.0
│
├── Data Layer          BBC News CSV → data/raw/ → data/processed/df_final.pkl
├── Analysis Engine     Preprocessing → TF-IDF → Sentiment → NER → Topics → Clusters
├── Classifier          Logistic Regression (saved to data/models/classifier.pkl)
├── LLM Layer           ollama (llama3.2:1b) / Gemini API (Colab) for generation
├── Multilingual        langdetect + deep-translator (Google Translate)
├── Conversation        Rule-based intent classifier + ArticleQueryEngine
└── Web Interface       Flask app (app.py) + HTML/CSS/JS frontend
```

---

## Repository Structure

```
ITAI2373-NewsBot-Final/
├── app.py                           ← Flask web application (run this)
├── requirements.txt                 ← All dependencies with versions
├── README.md                        ← This file
├── .gitignore
│
├── app/                             ← Flask frontend
│   ├── templates/
│   │   ├── base.html
│   │   └── index.html               ← Main dashboard UI
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── config/
│   ├── settings.py                  ← All hyperparameters and config
│   └── api_keys_template.txt        ← Copy to .env, never commit
│
├── src/                             ← NLP source modules
│   ├── data_processing/
│   │   ├── text_preprocessor.py     ← clean_text, NER, POS, tokenization
│   │   ├── feature_extractor.py     ← TF-IDF, Count vectorizers, custom features
│   │   └── data_validator.py        ← Data quality checks
│   ├── analysis/
│   │   ├── classifier.py            ← NewsClassifier class
│   │   ├── sentiment_analyzer.py    ← VADER sentiment
│   │   ├── ner_extractor.py         ← Named entity recognition
│   │   └── topic_modeler.py         ← TopicModeler class (LDA + NMF)
│   ├── language_models/
│   │   ├── summarizer.py            ← generate_summary() via ollama
│   │   ├── generator.py             ← enhance_content(), generate_insights()
│   │   └── embeddings.py            ← SemanticSearchEngine
│   ├── multilingual/
│   │   ├── language_detector.py     ← detect_language() via langdetect
│   │   ├── translator.py            ← translate_text() via deep-translator
│   │   └── cross_lingual_analyzer.py
│   ├── conversation/
│   │   ├── query_processor.py       ← QueryProcessor, intent classification
│   │   ├── intent_classifier.py     ← IntentClassifier (rule + embedding)
│   │   └── response_generator.py   ← ArticleQueryEngine (multi-turn Q&A)
│   └── utils/
│       ├── visualization.py         ← Reusable matplotlib plots
│       ├── evaluation.py            ← Classification + clustering metrics
│       └── export.py                ← Report generation
│
├── notebooks/                       ← Run in order on Google Colab
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Advanced_Classification.ipynb   ← Trains + saves classifier
│   ├── 03_Topic_Modeling.ipynb            ← Module A (LDA + NMF)
│   ├── 04_Language_Models.ipynb           ← Module B (LLM pipeline)
│   ├── 05_Multilingual_Analysis.ipynb     ← Module C
│   ├── 06_Conversational_Interface.ipynb  ← Module D
│   └── 07_System_Integration.ipynb        ← End-to-end pipeline
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   ├── test_topic_modeling.py
│   └── test_integration.py
│
├── data/
│   ├── raw/           ← BBC News CSV (download via Kaggle, not committed)
│   ├── processed/     ← df_final.pkl (generated by notebooks)
│   ├── models/        ← classifier.pkl, tfidf_vectorizer.pkl (generated by notebook 02)
│   └── results/       ← Analysis outputs, visualizations
│
└── docs/
    ├── technical_documentation.md
    ├── user_guide.md
    ├── api_reference.md
    ├── deployment_guide.md
    └── individual_contributions.md
```

---

## Setup & Installation

### Requirements
- Python 3.10+
- [ollama](https://ollama.ai) (for local LLM features)
- Google Colab or Jupyter (for notebooks)

### Quick setup script

```bash
# Clone and enter project


# Create virtual environment
python -m venv newsbot_env

# Activate (Mac/Linux)
source newsbot_env/bin/activate
# Activate (Windows Git Bash)
source newsbot_env/Scripts/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy API key template
cp config/api_keys_template.txt .env
# Edit .env with your values
```

### Download the BBC Dataset

```bash
# Option A: Kaggle CLI
kaggle datasets download -d shivamkushwaha/bbc-full-text-document-classification \
    --unzip -p data/raw/

# Option B: Manual
# Download from https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification
# Place the CSV in data/raw/
```

### Train the models (Google Colab recommended)

Run notebooks in order — each saves its outputs to Google Drive:

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

After notebook 02 completes, download from Google Drive:
- `data/models/classifier.pkl`
- `data/models/tfidf_vectorizer.pkl`
- `data/models/count_vectorizer.pkl`
- `data/models/model_info.json`
- `data/processed/df_final.pkl`

Place them in the corresponding local folders.

### Run the web app

```bash
# Start ollama in a separate terminal (for LLM features)
ollama serve

# Start Flask app
python app.py
# Open: http://127.0.0.1:5000
```

---

## Web Application Features

The Flask app (`app.py`) exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/analyze` | POST | Full NLP pipeline + trained classifier inference |
| `/summarize` | POST | Abstractive summarization via LLM |
| `/translate` | POST | Translation + language detection |
| `/query` | POST | Corpus-level natural language query |
| `/chat` | POST | Article Q&A with session history |
| `/reset_chat` | POST | Clear conversation history |
| `/enhance` | POST | Contextual content enrichment |
| `/insights` | POST | Structured insight generation |
| `/health` | GET | System status JSON |

### LLM Configuration

The app supports two LLM backends — set in `config/settings.py`:

```python
# Local (default)
OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"

# Or use Gemini in Colab notebooks:
# GEMINI_MODEL = "gemini-2.0-flash"
```

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `spacy` | Tokenization, NER, POS tagging |
| `vaderSentiment` | Sentiment analysis |
| `scikit-learn` | TF-IDF, LDA, NMF, Logistic Regression |
| `joblib` | Model persistence |
| `ollama` | Local LLM inference |
| `langdetect` | Language identification |
| `deep-translator` | Google Translate wrapper |
| `pyLDAvis` | Interactive topic visualization |
| `flask` | Web application framework |
| `matplotlib` / `seaborn` | Visualizations |

---

## Individual Contributions

See `docs/individual_contributions.md` for full breakdown.

**Jemima Egwurube** — All modules (A, B, C, D), Flask web app, classifier training,
topic modeling, LLM pipeline, multilingual analysis, conversational interface,
technical documentation, system integration.

---

## Academic Integrity

All core NLP implementations are original work. External libraries are attributed in
`requirements.txt`. AI assistance (Claude) used for scaffolding and debugging,
documented per course policy.

---

## Course Information

- **Course:** ITAI 2373 — Natural Language Processing
- **Institution:** Houston Community College
- **Semester:** Spring 2025
- **Dataset:** [BBC News Dataset](https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification) (Kaggle)
