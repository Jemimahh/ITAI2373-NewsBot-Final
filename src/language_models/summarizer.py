"""
src/language_models/summarizer.py
Abstractive summarization via ollama/llama3.2 — Module B.1.

Provides generate_summary() and batch_summarize() functions.
"""

import logging
from config.settings import OLLAMA_MODEL, OLLAMA_HOST, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str, system: str = None) -> str:
    """
    Low-level wrapper around the ollama Python client.

    Args:
        prompt: User prompt.
        system: Optional system instruction.

    Returns:
        Model response string.

    Raises:
        RuntimeError: If ollama is not running or model unavailable.
    """
    try:
        import ollama
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
        )
        return response["message"]["content"].strip()
    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")
    except Exception as e:
        raise RuntimeError(f"ollama call failed: {e}\nEnsure ollama is running: ollama serve")


def generate_summary(
    article_text: str,
    max_sentences: int = 3,
    preserve_entities: bool = True,
    category: str = None,
) -> dict:
    """
    Generate an abstractive summary of a news article.

    Uses llama3.2 via ollama to produce a concise summary that:
    - Preserves Who/What/When/Where/Why (5W framework)
    - Maintains named entities if preserve_entities=True
    - Targets max_sentences length

    Args:
        article_text: Full article text.
        max_sentences: Target summary length in sentences.
        preserve_entities: If True, instruct model to keep entity names exact.
        category: Optional category hint (e.g. "tech") for context.

    Returns:
        Dict with: summary (str), word_count (int), compression_ratio (float).
    """
    system = (
        "You are a professional news editor. Write accurate, concise summaries. "
        "Never add information not present in the source article. "
        "Preserve all named entities (people, organizations, locations) exactly as written."
        if preserve_entities else
        "You are a professional news editor. Write accurate, concise summaries."
    )

    cat_hint = f" This is a {category} news article." if category else ""
    prompt = (
        f"Summarize the following news article in exactly {max_sentences} sentences.{cat_hint} "
        "Cover Who, What, When, Where, and Why (5W) where applicable. "
        "Output only the summary, no preamble.\n\n"
        f"ARTICLE:\n{article_text[:3000]}"
    )

    summary = _call_ollama(prompt, system)

    original_words = len(article_text.split())
    summary_words  = len(summary.split())
    compression    = round(1 - summary_words / max(original_words, 1), 3)

    return {
        "summary":          summary,
        "word_count":       summary_words,
        "compression_ratio": compression,
    }


def batch_summarize(
    df,
    text_col: str = "cleaned_text",
    category_col: str = "category",
    max_sentences: int = 3,
    n_samples: int = None,
) -> list[dict]:
    """
    Run generate_summary on multiple articles.

    Args:
        df: DataFrame with article text.
        text_col: Column name for article text.
        category_col: Column name for category.
        max_sentences: Target sentence count per summary.
        n_samples: If set, only process first n_samples rows.

    Returns:
        List of summary result dicts.
    """
    from tqdm import tqdm

    subset = df if n_samples is None else df.head(n_samples)
    results = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Summarizing"):
        try:
            result = generate_summary(
                article_text=row[text_col],
                category=row.get(category_col),
                max_sentences=max_sentences,
            )
            result["category"] = row.get(category_col)
            results.append(result)
        except Exception as e:
            logger.warning(f"Summary failed for row {_}: {e}")
            results.append({"summary": None, "error": str(e)})

    return results
