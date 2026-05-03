"""
src/utils/export.py
Report and export utilities for NewsBot 2.0 analysis results.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from config.settings import RESULTS_DIR

logger = logging.getLogger(__name__)


def export_dataframe(df: pd.DataFrame, filename: str, fmt: str = "csv") -> Path:
    """
    Export a DataFrame to CSV or JSON.

    Args:
        df: DataFrame to export.
        filename: Output filename (without extension).
        fmt: 'csv' or 'json'.

    Returns:
        Path to the saved file.
    """
    path = RESULTS_DIR / f"{filename}.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    logger.info(f"Exported {len(df)} rows to {path}")
    return path


def export_topic_report(topics: dict, labels: dict, filename: str = "topic_report") -> Path:
    """
    Export a human-readable topic modeling report as JSON.

    Args:
        topics: Dict of topic_id → [(word, weight), ...].
        labels: Dict of topic_id → label string.
        filename: Output filename.

    Returns:
        Path to saved JSON file.
    """
    report = {
        str(t_id): {
            "label":     labels.get(t_id, f"Topic {t_id}"),
            "top_words": [{"word": w, "weight": round(s, 4)} for w, s in words[:15]],
        }
        for t_id, words in topics.items()
    }
    path = RESULTS_DIR / f"{filename}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info(f"Topic report saved: {path}")
    return path


def generate_system_summary(df: pd.DataFrame, modeler=None, metrics: dict = None) -> str:
    """
    Generate a plain-text system summary report.

    Args:
        df: Processed DataFrame (df_final).
        modeler: Optional fitted TopicModeler instance.
        metrics: Optional dict of evaluation metrics.

    Returns:
        Multi-line summary string.
    """
    lines = [
        "=" * 60,
        "   NEWSBOT INTELLIGENCE SYSTEM 2.0 — ANALYSIS REPORT",
        "=" * 60,
        f"   Dataset       : BBC News ({len(df)} articles)",
        f"   Categories    : {sorted(df['category'].unique().tolist())}",
    ]

    if "sentiment_label" in df.columns:
        dist = df["sentiment_label"].value_counts().to_dict()
        lines.append(f"   Sentiment     : {dist}")

    if modeler and modeler._is_fitted:
        lines.append(f"   Topics        : {modeler.n_topics} (LDA + NMF)")

    if metrics:
        for k, v in metrics.items():
            lines.append(f"   {k:<20}: {v}")

    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)

    path = RESULTS_DIR / "system_summary.txt"
    path.write_text(report)
    return report
