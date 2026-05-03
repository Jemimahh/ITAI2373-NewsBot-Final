# NewsBot 2.0 — API Reference

## src.data_processing.text_preprocessor

### `clean_text(text: str) -> str`
Apply baseline text cleaning: expand contractions, lowercase, remove URLs/HTML/special chars.

### `tokenize_and_process(text, remove_stopwords=True, lemmatize=True) -> list[str]`
Tokenize using spaCy with optional stop-word removal and lemmatization.

### `extract_named_entities(text: str) -> dict[str, list[str]]`
Extract named entities grouped by type (PERSON, ORG, GPE, LOC, etc.).

### `preprocess_dataframe(df, text_col='text', label_col='category') -> DataFrame`
Apply full pipeline to a DataFrame. Adds: `cleaned_text`, `tokens`, `entities`, `pos_tags`.

---

## src.analysis.topic_modeler.TopicModeler

### `__init__(n_topics=10, method='both')`
Initialize with topic count and method ('lda', 'nmf', or 'both').

### `fit(count_matrix, tfidf_matrix, count_vocab, tfidf_vocab) -> self`
Train topic model(s). Returns self for chaining.

### `get_topic_words(model='nmf', n_words=10) -> dict`
Returns `{topic_id: [(word, weight), ...]}`.

### `get_dominant_topics(model='nmf') -> np.ndarray`
Returns array of dominant topic index per document.

### `auto_label_topics(model='nmf') -> dict`
Returns `{topic_id: label_string}` auto-generated from top words.

### `cluster_documents(k=None) -> np.ndarray`
K-Means cluster on NMF vectors. Auto-selects k if None.

### `visualize_topics(model='nmf', n_words=10, save_path=None) -> None`
Plot topic word bar charts. LDA: pyLDAvis HTML. NMF: matplotlib grid.

### `get_evaluation_metrics(count_matrix) -> dict`
Returns perplexity (LDA), reconstruction_error (NMF).

---

## src.analysis.classifier.NewsClassifier

### `__init__(model_type='logreg')`
Options: 'logreg' (Logistic Regression), 'svm' (Linear SVC with Platt calibration).

### `fit(X, y) -> self`
Train on feature matrix X and label array y.

### `predict(X) -> np.ndarray`
Predict category labels.

### `predict_with_confidence(X) -> tuple[np.ndarray, np.ndarray]`
Returns (labels, confidence_scores).

### `evaluate(X, y_true) -> dict`
Returns classification report string and confusion matrix.

### `cross_validate(X, y, cv=5) -> dict`
Returns `{mean_accuracy, std_accuracy}` from stratified k-fold CV.

---

## src.language_models.summarizer

### `generate_summary(article_text, max_sentences=3, preserve_entities=True, category=None) -> dict`
Returns `{summary, word_count, compression_ratio}`.

### `batch_summarize(df, text_col, category_col, max_sentences, n_samples) -> list[dict]`
Batch summarization over a DataFrame.

---

## src.language_models.generator

### `enhance_content(article_text, category=None, entities=None) -> dict`
Returns `{background_context, related_trends, implications, entities_to_watch}`.

### `generate_insights(article_text, nlp_metadata=None) -> dict`
Returns `{key_findings, patterns, entities_of_interest, sentiment_drivers, anomalies, recommended_queries}`.

---

## src.multilingual

### `detect_language(text: str) -> dict`
Returns `{language_code, language_name, confidence, all_detections}`.

### `translate_text(text, target_lang='en', source_lang='auto') -> dict`
Returns `{translated_text, source_lang, target_lang, char_count}`.

### `cross_lingual_sentiment(article_text, languages=None) -> dict`
Returns sentiment analysis per language after translation round-trip.

---

## src.conversation.query_processor.QueryProcessor

### `__init__(df=None)`
Initialize with optional DataFrame for query execution.

### `classify_intent(query: str) -> str`
Returns intent string: filter_by_sentiment, filter_by_category, summarize, etc.

### `extract_filters(query: str) -> dict`
Returns `{category, sentiment_label, keyword}` extracted from query.

### `process(query: str) -> dict`
Full query execution: returns `{intent, filters, n_results, response_text, data}`.

---

## src.conversation.response_generator.ArticleQueryEngine

### `__init__(article_text, category=None, metadata=None)`
Initialize with article text and optional NLP metadata.

### `ask(question: str) -> str`
Ask a question. Maintains conversation history across calls.

### `reset() -> None`
Clear conversation history.

### `show_history() -> None`
Print full conversation history to stdout.

### `history -> list[dict]`
Property: returns conversation history as `[{question, answer}, ...]`.
