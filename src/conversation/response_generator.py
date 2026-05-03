"""
src/conversation/response_generator.py
Context-aware response generation for the NewsBot chat interface.
Combines structured query results with LLM-generated explanations.
"""

import logging
from src.language_models.summarizer import _call_ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NewsBot 2.0, an expert AI news analyst. "
    "Answer questions about news articles clearly and concisely. "
    "If you don't know something, say so honestly. "
    "Always ground your responses in the provided article context."
)


class ArticleQueryEngine:
    """
    Stateful multi-turn Q&A engine for a single article.

    Maintains conversation history across turns so follow-up
    questions carry full context.

    Usage::

        engine = ArticleQueryEngine(article_text, category='tech')
        answer = engine.ask("What is the main claim in this article?")
        answer = engine.ask("What are the implications?")
        engine.show_history()
    """

    def __init__(self, article_text: str, category: str = None, metadata: dict = None):
        """
        Args:
            article_text: Full article text to analyze.
            category: BBC category (e.g. "tech").
            metadata: Optional pre-computed NLP metadata dict.
        """
        self.article_text = article_text
        self.category     = category
        self.metadata     = metadata or {}
        self._history: list[dict] = []

        meta_str = ""
        if metadata:
            parts = []
            if "sentiment_label" in metadata:
                parts.append(f"Sentiment: {metadata['sentiment_label']}")
            if "lda_topic_label" in metadata:
                parts.append(f"Topic: {metadata['lda_topic_label']}")
            meta_str = " | ".join(parts)

        cat_str = f"[{category.upper()}] " if category else ""
        meta_block = f"\nNLP Analysis: {meta_str}" if meta_str else ""

        self._system_context = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Article context: {cat_str}{meta_block}\n\n"
            f"ARTICLE:\n{article_text[:3000]}"
        )

    def ask(self, question: str) -> str:
        """
        Ask a question about the article. Maintains conversation context.

        Args:
            question: User's natural language question.

        Returns:
            LLM-generated answer string.
        """
        history_str = ""
        if self._history:
            history_str = "\n".join(
                f"Q: {turn['question']}\nA: {turn['answer']}"
                for turn in self._history[-4:]  # last 4 turns for context
            )
            history_str = f"\nConversation history:\n{history_str}\n"

        prompt = f"{self._system_context}{history_str}\nQuestion: {question}\nAnswer:"

        try:
            answer = _call_ollama(prompt)
        except Exception as e:
            answer = f"I encountered an error: {e}"
            logger.error(f"ArticleQueryEngine.ask failed: {e}")

        self._history.append({"question": question, "answer": answer})
        return answer

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []
        logger.info("Conversation history cleared.")

    def show_history(self) -> None:
        """Print the full conversation history."""
        if not self._history:
            print("No conversation history yet.")
            return
        print("=" * 55)
        print("  CONVERSATION HISTORY")
        print("=" * 55)
        for i, turn in enumerate(self._history, 1):
            print(f"\n[Q{i}] {turn['question']}")
            print(f"[A{i}] {turn['answer']}")
        print("=" * 55)

    @property
    def history(self) -> list[dict]:
        """Return conversation history as list of dicts."""
        return self._history.copy()


class ResponseGenerator:
    """
    Generate natural language responses to structured query results.

    Wraps ArticleQueryEngine for corpus-level conversational queries.
    """

    def __init__(self, df=None):
        """
        Args:
            df: Full DataFrame for corpus-level queries.
        """
        self.df = df

    def generate(self, query_result: dict) -> str:
        """
        Turn a structured query result into a natural language response.

        Args:
            query_result: Dict from QueryProcessor.process().

        Returns:
            Human-readable response string.
        """
        if "error" in query_result:
            return query_result["error"]

        intent = query_result.get("intent", "general")
        n      = query_result.get("n_results", 0)
        text   = query_result.get("response_text", "")

        if intent == "filter_by_category":
            cat = query_result.get("filters", {}).get("category", "that category")
            return (
                f"I found {n} articles in the {cat} category. "
                f"{text} Would you like me to summarize the top articles or analyze their sentiment?"
            )

        if intent == "filter_by_sentiment":
            return f"{text} Would you like to explore what's driving this sentiment?"

        if intent == "stats":
            return f"{text} I can break this down further by topic or entity if you'd like."

        return text if text else f"I found {n} matching articles."
