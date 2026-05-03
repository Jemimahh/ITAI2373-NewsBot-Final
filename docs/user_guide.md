# NewsBot 2.0 — User Guide

## What is NewsBot 2.0?

NewsBot Intelligence System 2.0 is an AI-powered news analysis platform that
automatically processes BBC news articles to extract topics, sentiment, named
entities, and generate intelligent summaries and answers.

## Getting Started

### Option A: Run Notebooks in Google Colab (Recommended)

1. Open the notebooks folder in this repository
2. Upload them to Google Colab (File → Upload Notebook)
3. Run the setup cell in each notebook first to install dependencies
4. Run cells top-to-bottom — each notebook builds on the previous one

**Notebook order:**
```
01_Data_Exploration        → understand the BBC dataset
02_Advanced_Classification → classify and evaluate the model
03_Topic_Modeling          → discover hidden topics (Module A)
04_Language_Models         → LLM summarization and Q&A (Module B)
05_Multilingual_Analysis   → translate and compare (Module C)
06_Conversational_Interface → chat with your dataset (Module D)
07_System_Integration      → full pipeline end-to-end
```

### Option B: Run the Web Application

1. Open `NewsBot_IntelligenceSystem_2.html` in any modern browser
2. Select a sample article from the left panel
3. View NLP analysis results in the left panel
4. Use the four tabs on the right for AI-powered features:
   - **Summarize** — get a concise article summary
   - **Enhance** — see background context and implications
   - **Query** — ask questions about the article
   - **Insights** — get structured findings and recommendations

> Note: The web app uses the Claude API. It works automatically inside Claude.ai.
> For standalone use, enter your Anthropic API key when prompted.

---

## Feature Guide

### Topic Modeling (Module A)

**What it does:** Discovers hidden themes across 2,225 BBC articles using two
complementary algorithms — LDA and NMF.

**How to interpret results:**
- Each topic is labeled with its top keywords (e.g. "Technology / Software / Apple")
- The **evolution heatmap** shows which topics dominate each category
- The **cluster scatter plot** shows articles grouped by topic similarity

**Tip:** Articles in overlapping cluster regions have mixed themes (e.g. tech-business crossover).

### Summarization (Module B.1)

**What it does:** Generates a 3-sentence abstractive summary that captures Who, What, When,
Where, and Why.

**How to use it:**
- In notebooks: `generate_summary(article_text, max_sentences=3)`
- In the web app: click the **Summarize** tab

### Content Enhancement (Module B.2)

Provides three enrichment layers:
- **Background Context** — historical or domain context for the story
- **Related Trends** — broader patterns the story connects to
- **Implications** — potential consequences or significance

### Article Q&A (Module B.3 / Module D)

Maintains conversation history so follow-up questions work naturally:
```
Q: "What is the main claim?"
Q: "Who are the key people mentioned?"
Q: "What might happen next?"   ← uses previous answers as context
```

### Multilingual Analysis (Module C)

**Supported languages:** English, French, German, Spanish, Portuguese, Italian,
Arabic, Chinese, Japanese, Russian.

**How to translate an article:**
```python
from src.multilingual.translator import translate_text
result = translate_text(article_text, target_lang="fr")
print(result["translated_text"])
```

---

## Frequently Asked Questions

**Q: Do I need an internet connection?**
A: For the web app and cloud notebooks, yes. For ollama/LLM features, no —
ollama runs locally on your machine.

**Q: The LLM isn't responding — what's wrong?**
A: Make sure ollama is running: open a terminal and run `ollama serve`.
Also ensure the model is downloaded: `ollama pull llama3.2`.

**Q: The sentiment seems wrong for an article.**
A: VADER is a lexicon-based analyzer optimized for short social media text.
For nuanced long-form news, sentiment may not always match human judgment.
This is a known limitation discussed in the Technical Documentation.

**Q: How do I add my own articles?**
A: Load a CSV with `text` and `category` columns and pass it to the preprocessing
pipeline. See `01_Data_Exploration.ipynb` for the exact data loading steps.

**Q: What languages does language detection support?**
A: `langdetect` supports 55 languages. The translation feature supports 10 target
languages configured in `config/settings.py`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: spacy` | Run `pip install -r requirements.txt` |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run `python -m spacy download en_core_web_sm` |
| `ConnectionError: ollama` | Run `ollama serve` in a separate terminal |
| pyLDAvis not displaying | Save as HTML and open in browser: `lda_interactive.html` |
| Kaggle dataset not found | Download manually from Kaggle and place in `data/raw/` |
