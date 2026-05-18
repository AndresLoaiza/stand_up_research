"""
Precompute everything the dashboard reads so each render is just a
parquet lookup. Run after changing transcripts, cleaning rules, or
the curated topic lexicons.

Writes to data/data_frame/:
  - df_enriched.parquet : df_unified columns + sentiment, ttr,
    word_count, plus one column per NRC emotion (nrc_<emotion>).
  - emotion_words.parquet : long-form [show_idx, emotion, word, count]
    for the "top words per emotion" panel.
  - topic_scores.parquet : long-form [show_idx, topic_id, score]
    (hits per 1000 tokens) for each curated theme.

Run from project root:
    python analysis/precompute.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    DEFAULT_STOPWORDS,
    EMOTIONS,
    load_unified,
    sentiment_compound,
    tokenize,
)
from analysis.compute_curated_topics import (  # noqa: E402
    THEMES,
    score_show,
    tokenize_for_scoring,
)

DF_DIR = ROOT / "data" / "data_frame"
ENRICHED = DF_DIR / "df_enriched.parquet"
EMO_WORDS = DF_DIR / "emotion_words.parquet"
TOPIC_SCORES = DF_DIR / "topic_scores.parquet"


def compute_nrc_per_show(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run NRC once per show. Returns:
       - profile_df : index aligned to df.index with one column per emotion (proportions).
       - words_df   : long-form [show_idx, emotion, word, count] aggregated per show.
    """
    from nrclex import NRCLex

    profile_rows = []
    word_rows = []
    for idx, row in df.iterrows():
        nrc = NRCLex()
        nrc.load_raw_text(row["transcript"] or "x")
        freq = nrc.affect_frequencies
        profile_rows.append({"show_idx": idx, **{e: float(freq.get(e, 0.0)) for e in EMOTIONS}})

        # Build the per-emotion word counter for this show
        counters: dict[str, Counter] = {e: Counter() for e in EMOTIONS}
        for w in nrc.words:
            wl = w.lower()
            for emo in nrc.affect_dict.get(wl, []):
                if emo in counters:
                    counters[emo][wl] += 1
        for emo, c in counters.items():
            for w, n in c.items():
                word_rows.append({"show_idx": idx, "emotion": emo, "word": w, "count": n})

    profile = pd.DataFrame(profile_rows).set_index("show_idx")
    profile.columns = [f"nrc_{c}" for c in profile.columns]
    return profile, pd.DataFrame(word_rows)


def compute_topic_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        counter = tokenize_for_scoring(row["transcript"])
        total = sum(counter.values())
        for theme in THEMES:
            sc = score_show(counter, theme["lexicon"], total)
            rows.append({"show_idx": idx, "topic_id": theme["id"], "score": sc})
    return pd.DataFrame(rows)


def main():
    print("Loading df_unified...")
    df = load_unified()
    print(f"  {len(df)} shows.")

    # 1. Sentiment + lexical stats (cheap)
    print("Computing sentiment + lexical stats...")
    df["sentiment"] = df["transcript"].map(sentiment_compound)
    tokens = df["transcript"].map(lambda t: tokenize(t))
    df["unique_words"] = tokens.map(lambda t: len(set(t)))
    df["ttr"] = df["unique_words"] / tokens.map(len).replace(0, np.nan)
    df["words_per_min"] = df["word_count"] / df["runtime_min"]

    # 2. NRC profile + word counters
    print("Computing NRC profile + per-emotion word counters...")
    profile, emo_words = compute_nrc_per_show(df)
    df = df.join(profile, how="left")

    # 3. Curated topic scores
    print("Computing curated topic scores...")
    topic_scores = compute_topic_scores(df)

    # Persist (drop the bulky tokens column before saving)
    df_to_save = df.drop(columns=[c for c in df.columns
                                  if c in ("tokens",)], errors="ignore")
    df_to_save.to_parquet(ENRICHED)
    emo_words.to_parquet(EMO_WORDS)
    topic_scores.to_parquet(TOPIC_SCORES)

    print(f"\nWrote:")
    print(f"  {ENRICHED}  ({len(df_to_save)} rows, {len(df_to_save.columns)} cols)")
    print(f"  {EMO_WORDS}  ({len(emo_words)} rows)")
    print(f"  {TOPIC_SCORES}    ({len(topic_scores)} rows)")


if __name__ == "__main__":
    main()
