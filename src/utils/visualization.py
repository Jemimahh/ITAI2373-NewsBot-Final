"""
src/utils/visualization.py
Reusable plotting utilities for NewsBot 2.0.
All functions return Matplotlib figures for flexibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from config.settings import PLOT_DPI, PLOT_FACECOLOR, PLOT_PALETTE

_STYLE = {
    "figure.facecolor": PLOT_FACECOLOR,
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2d3561",
    "axes.labelcolor":  "#e8d5a3",
    "xtick.color":      "#a0a8c0",
    "ytick.color":      "#a0a8c0",
    "text.color":       "#e8d5a3",
    "grid.color":       "#2d3561",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
    "figure.dpi":       PLOT_DPI,
}


def apply_style():
    """Apply NewsBot dark theme to matplotlib."""
    plt.rcParams.update(_STYLE)


def plot_category_distribution(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Bar chart of article count per BBC category."""
    apply_style()
    cat_counts = df["category"].value_counts()
    colors = [PLOT_PALETTE.get(c, "#ffffff") for c in cat_counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, alpha=0.9, width=0.6)

    for bar, count in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("BBC News Dataset — Category Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Article Count", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI, facecolor=PLOT_FACECOLOR)
    return fig


def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Stacked bar chart of sentiment labels per category."""
    apply_style()
    if "sentiment_label" not in df.columns:
        raise ValueError("DataFrame must have 'sentiment_label' column.")

    ct = pd.crosstab(df["category"], df["sentiment_label"], normalize="index")
    sentiment_colors = {"Positive": "#81c784", "Neutral": "#e8d5a3", "Negative": "#e57373"}

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = np.zeros(len(ct))
    for label in ["Positive", "Neutral", "Negative"]:
        if label in ct.columns:
            ax.bar(ct.index, ct[label], bottom=bottoms,
                   color=sentiment_colors[label], alpha=0.9, label=label, width=0.6)
            bottoms += ct[label].values

    ax.set_title("Sentiment Distribution by Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Proportion", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI, facecolor=PLOT_FACECOLOR)
    return fig


def plot_topic_heatmap(
    heat: np.ndarray,
    categories: list[str],
    n_topics: int,
    title: str = "Topic Distribution",
    save_path: str = None,
) -> plt.Figure:
    """Category × topic heatmap."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f"T{i}" for i in range(n_topics)], fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.upper() for c in categories], fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Proportion", shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI, facecolor=PLOT_FACECOLOR)
    return fig


def plot_entity_frequency(entity_freq: list[tuple], top_n: int = 20, save_path: str = None) -> plt.Figure:
    """Horizontal bar chart of top entity frequencies."""
    apply_style()
    entities = [e for e, _ in entity_freq[:top_n]][::-1]
    counts   = [c for _, c in entity_freq[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(entities, counts, color="#4fc3f7", alpha=0.85)
    ax.set_title(f"Top {top_n} Most Frequent Named Entities", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=PLOT_DPI, facecolor=PLOT_FACECOLOR)
    return fig
