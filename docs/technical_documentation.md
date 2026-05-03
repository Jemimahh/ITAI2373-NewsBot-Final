# NewsBot 2.0 — Technical Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  NewsBot Intelligence System 2.0            │
├─────────────────────────────────────────────────────────────┤
│  INPUT LAYER                                                │
│  BBC News CSV (2,225 articles, 5 categories)                │
├─────────────────────────────────────────────────────────────┤
│  DATA PROCESSING  (src/data_processing/)                    │
│  text_preprocessor → feature_extractor → data_validator     │
│  [contractions, clean_text, spaCy tokenize, NER, POS]       │
├─────────────────────────────────────────────────────────────┤
│  ANALYSIS ENGINE  (src/analysis/)                           │
│  classifier → sentiment_analyzer → ner_extractor           │
│  topic_modeler [LDA + NMF + K-Means]                        │
├─────────────────────────────────────────────────────────────┤
│  LANGUAGE MODELS  (src/language_models/)                    │
│  summarizer → generator → embeddings                        │
│  [ollama/llama3.2 local inference]                          │
├─────────────────────────────────────────────────────────────┤
│  MULTILINGUAL     (src/multilingual/)                       │
│  language_detector → translator → cross_lingual_analyzer    │
│  [langdetect + deep-translator/Google]                      │
├─────────────────────────────────────────────────────────────┤
│  CONVERSATION     (src/conversation/)                       │
│  query_processor → intent_classifier → response_generator   │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT LAYER                                               │
│  Web App (HTML) | Notebooks | CSV/JSON exports | Plots      │
└─────────────────────────────────────────────────────────────┘
```

## Module A — Topic Modeling

### Algorithm: Latent Dirichlet Allocation (LDA)

LDA models each document as a mixture of K topics, where each topic is a
probability distribution over vocabulary words. The generative process:

1. For each document d: sample topic distribution θ_d ~ Dirichlet(α)
2. For each word position in d: sample topic z ~ Categorical(θ_d)
3. Sample word w ~ Categorical(φ_z), where φ_z is the topic-word distribution

**Key hyperparameters (config/settings.py):**
- `N_TOPICS = 10` — number of latent topics
- `LDA_DOC_PRIOR = 0.1` — sparse document-topic distributions
- `LDA_WORD_PRIOR = 0.01` — sparse topic-word distributions
- `LDA_MAX_ITER = 25` — online VB iterations

**Evaluation metric:** Perplexity (lower = better fit on held-out data)

### Algorithm: Non-negative Matrix Factorization (NMF)

NMF factorizes the TF-IDF matrix V ≈ WH where:
- W (n_docs × n_topics) = document-topic weights
- H (n_topics × n_terms) = topic-word weights

All entries are non-negative, yielding parts-based, interpretable representations.

**Key hyperparameters:**
- `init = 'nndsvda'` — deterministic SVD-based initialization
- `NMF_ALPHA = 0.1` — L1/L2 regularization
- `NMF_L1_RATIO = 0.5` — Elastic Net mixing

**Evaluation metric:** Frobenius reconstruction error ||V - WH||_F

### Content Clustering

K-Means clusters documents using NMF topic weight vectors (normalized to unit L1 norm)
as feature representations. Optimal k selected via silhouette score over range [2, 12].

## Module B — Language Model Integration

### LLM Backend: ollama + llama3.2

All LLM calls route through `src/language_models/summarizer._call_ollama()`:
- Model: `llama3.2` (3.2B parameter instruction-tuned model)
- Inference: local, via ollama HTTP API at `localhost:11434`
- No external API calls or costs
- Temperature: 0.3 (low for factual news tasks)
- Max tokens: 512 per call

### Summarization (B.1)
Prompt strategy: 5W framework (Who/What/When/Where/Why) with entity preservation
constraint. Compression ratio = 1 − (summary_words / article_words).

### Content Enhancement (B.2)
JSON-mode output with three enrichment layers. Falls back to raw text if JSON
parse fails. Entity list from Module 8 NER is injected into prompt to reduce
hallucination of invented entity names.

### Insight Generation (B.4)
NLP metadata (VADER compound, top TF-IDF terms, topic labels) is serialized
into the prompt to ground the LLM analysis in quantitative signals.

## Module C — Multilingual Intelligence

### Language Detection
`langdetect` uses n-gram frequency profiles (based on Nakatani Shuyo's language
detection library). `DetectorFactory.seed = 42` ensures reproducible results.
Detection confidence < 0.85 should be treated with caution on short texts.

### Translation
`deep-translator` wraps the Google Translate web API (no key required for
moderate usage). Texts > 4,500 chars are chunked at sentence boundaries.
For production: replace with DeepL API (`DEEPL_API_KEY` in config).

## Module D — Conversational Interface

### Intent Classification
Hybrid approach:
1. **Rule-based (primary):** Regex pattern matching over 8 intent categories
2. **Embedding fallback:** Sentence-transformer similarity to curated intent examples

### Context Management
`ArticleQueryEngine` maintains conversation history as a list of `{question, answer}`
dicts. The last 4 turns are injected into each new prompt (sliding window).

## Data Flow

```
BBC CSV
  ↓ text_preprocessor.py
  cleaned_text, tokens, entities, pos_tags
  ↓ feature_extractor.py
  tfidf_matrix, count_matrix, custom_features
  ↓ sentiment_analyzer.py
  sentiment_compound, sentiment_label
  ↓ topic_modeler.py (TopicModeler.fit)
  lda_doc_topics_, nmf_doc_topics_, cluster
  ↓ df_final (all columns merged)
  ↓ language_models/ + multilingual/ + conversation/
  LLM responses, translations, query results
```

## Performance Considerations

- spaCy `en_core_web_sm` is used for speed; upgrade to `en_core_web_lg` for
  better NER accuracy at the cost of ~560MB RAM
- LDA with `learning_method='online'` is faster than batch on large corpora
- ollama inference speed depends on hardware; GPU acceleration is automatic
  if CUDA is available
- For datasets > 50,000 articles, consider FAISS approximate nearest neighbor
  search instead of exact cosine similarity in `SemanticSearchEngine`

## Known Limitations

1. **VADER sentiment** is lexicon-based and may misclassify irony or domain-specific
   language in financial and political reporting
2. **LDA topic interpretability** depends heavily on preprocessing quality;
   poor stop-word removal leads to generic topics
3. **Translation quality** via Google Translate may distort sentiment in
   the cross-lingual analysis
4. **llama3.2 hallucination** — LLM outputs should be treated as analysis aids,
   not authoritative fact sources
