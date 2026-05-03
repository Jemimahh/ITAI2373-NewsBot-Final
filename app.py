"""
app.py — NewsBot Intelligence System 2.0
Flask Web Application | ITAI 2373 Final Project

Run: python app.py
     open http://127.0.0.1:5000
"""

import sys, os, json, logging, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from config.settings import CATEGORIES, SUPPORTED_LANGUAGES, OLLAMA_MODEL

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "newsbot-dev-key-2024")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load trained models at startup ────────────────────────────────────────
MODELS_DIR = Path("data/models")
CLF, TFIDF, MODEL_INFO = None, None, {}

def load_trained_models():
    global CLF, TFIDF, MODEL_INFO
    try:
        CLF   = joblib.load(MODELS_DIR / "classifier.pkl")
        TFIDF = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        with open(MODELS_DIR / "model_info.json") as f:
            MODEL_INFO = json.load(f)
        logger.info(f"Trained models loaded — accuracy: {MODEL_INFO.get('accuracy')}")
    except FileNotFoundError:
        logger.warning("No trained models found in data/models/. Run notebook 02 first.")

load_trained_models()


# ── Lazy NLP loaders ──────────────────────────────────────────────────────
def _nlp():
    from src.data_processing.text_preprocessor import clean_text, extract_named_entities, get_pos_tags
    from src.analysis.sentiment_analyzer import analyze_sentiment
    return clean_text, extract_named_entities, get_pos_tags, analyze_sentiment

def _llm():
    from src.language_models.summarizer import generate_summary
    from src.language_models.generator import enhance_content, generate_insights
    from src.conversation.response_generator import ArticleQueryEngine
    return generate_summary, enhance_content, generate_insights, ArticleQueryEngine

def _multilingual():
    from src.multilingual.translator import translate_text
    from src.multilingual.language_detector import detect_language
    return translate_text, detect_language

def _query_processor():
    from src.conversation.query_processor import QueryProcessor
    return QueryProcessor


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           categories=CATEGORIES,
                           languages=SUPPORTED_LANGUAGES,
                           model_info=MODEL_INFO)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Full NLP pipeline on submitted article text.
    Returns flat response fields that match the HTML frontend.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        clean_text, extract_named_entities, get_pos_tags, analyze_sentiment = _nlp()
        from collections import Counter

        cleaned   = clean_text(text)
        sentiment = analyze_sentiment(cleaned)
        raw_ents  = extract_named_entities(text)
        pos_tags  = get_pos_tags(text)
        pos_dist  = dict(Counter(tag for _, tag in pos_tags))
        words     = cleaned.split()
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Flatten entities → [{type, text}, ...] for the HTML tags
        entities_flat = [
            {"type": etype, "text": surface}
            for etype, surfaces in raw_ents.items()
            for surface in list(dict.fromkeys(surfaces))[:4]  # dedupe, max 4 per type
        ]

        # Trained classifier prediction
        predicted_category = ""
        confidence         = 0.0
        all_probs          = {}
        if CLF is not None and TFIDF is not None:
            X             = TFIDF.transform([cleaned])
            predicted_category = CLF.predict(X)[0]
            prob          = CLF.predict_proba(X)[0]
            confidence    = round(float(prob.max()), 4)
            all_probs     = {cat: round(float(p), 4)
                             for cat, p in zip(CLF.classes_, prob)}

        return jsonify({
            # Stats
            "word_count":           len(words),
            "sentence_count":       len(sentences),
            "avg_sent_len":         round(len(words) / max(len(sentences), 1), 1),
            # Flat sentiment fields (matches HTML)
            "sentiment_label":      sentiment["sentiment_label"],
            "sentiment_compound":   round(sentiment["compound"], 4),
            "positive":             round(sentiment["positive"], 4),
            "neutral":              round(sentiment["neutral"], 4),
            "negative":             round(sentiment["negative"], 4),
            # Entities as flat list
            "entities":             entities_flat,
            # POS
            "pos_distribution":     pos_dist,
            # Classifier
            "predicted_category":   predicted_category,
            "confidence":           confidence,
            "all_probabilities":    all_probs,
            "model_accuracy":       MODEL_INFO.get("accuracy", "N/A"),
        })
    except Exception as e:
        logger.error(f"/analyze error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        generate_summary, *_ = _llm()
        return jsonify(generate_summary(
            text,
            max_sentences=int(data.get("max_sentences", 3)),
            category=data.get("category", "")
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    """
    Translate article text. Returns flat fields that match the HTML.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        translate_text, detect_language = _multilingual()
        detection   = detect_language(text)
        translation = translate_text(text[:2000], target_lang=data.get("target_lang", "fr"))
        return jsonify({
            # Flat fields for the HTML
            "translated_text":   translation["translated_text"],
            "target_lang":       translation["target_lang"],
            "language_code":     detection["language_code"],
            "language":          detection["language_name"],
            "language_confidence": detection["confidence"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    Corpus-level natural language query using QueryProcessor.
    Matches the Query tab in the HTML (sends {query: str}).
    """
    data  = request.get_json(force=True)
    query_text = data.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    try:
        QueryProcessor = _query_processor()
        qp     = QueryProcessor()
        intent, conf = qp.classify_intent(query_text), 0.95
        intent = qp.classify_intent(query_text)
        filters = qp.extract_filters(query_text)
        return jsonify({
            "intent":   intent,
            "filters":  filters,
            "response": f"Query processed: intent={intent}, filters={filters}. "
                        f"Connect df_final to QueryProcessor for live corpus results.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Article-level Q&A using ArticleQueryEngine.
    Matches the Chat tab in the HTML (sends {question, article, category}).
    Maintains conversation history in the Flask session.
    """
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    article  = data.get("article", "").strip()
    category = data.get("category", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not article:
        return jsonify({"error": "Paste an article first, then ask questions about it."}), 400

    try:
        _, _, _, ArticleQueryEngine = _llm()

        # Restore history from session
        history = session.get("chat_history", [])

        engine = ArticleQueryEngine(article, category=category)
        for turn in history[-4:]:
            engine._history.append(turn)

        answer = engine.ask(question)

        # Save updated history to session
        session["chat_history"] = engine.history
        session.modified = True

        return jsonify({"answer": answer, "history": engine.history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    """Clear chat session history."""
    session.pop("chat_history", None)
    return jsonify({"status": "cleared"})


@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        _, enhance_content, *_ = _llm()
        return jsonify(enhance_content(
            text,
            category=data.get("category", ""),
            entities=data.get("entities", {})
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/insights", methods=["POST"])
def insights():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        _, _, generate_insights, _ = _llm()
        return jsonify(generate_insights(text, nlp_metadata=data.get("nlp_metadata", {})))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "model":         OLLAMA_MODEL,
        "categories":    CATEGORIES,
        "classifier":    MODEL_INFO.get("model_type", "not loaded"),
        "accuracy":      MODEL_INFO.get("accuracy", "N/A"),
    })


if __name__ == "__main__":
    print(f"\n  NewsBot 2.0 Flask App → http://127.0.0.1:5000")
    print(f"  Classifier: {MODEL_INFO.get('model_type','not loaded')} "
          f"(accuracy: {MODEL_INFO.get('accuracy','N/A')})\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
