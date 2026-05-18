"""
Reusable text-mining functions for stand-up comedy transcripts.

Used by both notebooks/analysis.ipynb and dashboard/app.py so the logic
stays in one place.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_PATH = PROJECT_ROOT / "data" / "data_frame" / "df_unified.parquet"

TOKEN_RE = re.compile(r"[a-z]+")

DEFAULT_STOPWORDS = {
    # NLTK english stopwords + colloquial speech fillers that dominate transcripts
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now",
    # speech-specific fillers
    "im", "youre", "thats", "dont", "didnt", "doesnt", "ive", "youve",
    "hes", "shes", "theyre", "weve", "wouldnt", "couldnt", "shouldnt",
    "wont", "cant", "isnt", "arent", "wasnt", "werent", "gonna", "wanna",
    "gotta", "yeah", "okay", "ok", "uh", "um", "like", "know", "right",
    "well", "one", "get", "got", "going", "go", "say", "said", "really",
    "people", "think", "thing", "things", "make", "made", "way", "would",
    "could", "still", "yes", "even", "much", "many", "guy", "guys",
    # extra noise common in stand-up transcripts
    "want", "wants", "wanted", "see", "seen", "saw", "looking", "look", "looks",
    "tell", "told", "telling", "ask", "asked", "asks", "give", "gives", "gave",
    "let", "lets", "put", "take", "takes", "took", "taken", "come", "comes", "came",
    "back", "anything", "everything", "nothing", "something",
    "everybody", "everyone", "nobody", "noone", "somebody", "someone",
    "actually", "literally", "kind", "kinda", "sort", "stuff",
    "always", "never", "ever", "anyway", "anyways",
    "first", "second", "next", "last", "another", "old", "new", "good", "bad",
    "big", "little", "small", "high", "low", "long", "short", "real",
    "today", "yesterday", "tomorrow", "night", "day", "time", "times", "year",
    "years", "minute", "minutes", "hour", "hours", "moment", "while",
    "started", "start", "stop", "stopped", "keep", "kept", "try", "tried", "trying",
    "feel", "felt", "feels", "feeling", "love", "hate", "called", "calls",
    "yeah", "uh-huh", "huh", "ah", "oh", "hey", "hi", "hello", "wow", "fuck",
    "fucking", "fucked", "shit", "damn", "ass", "bitch", "bullshit",
    "alright", "everybody", "everyone", "nobody", "anybody", "someone",
    "anyone", "anything", "everything", "nothing", "something",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "million",
    # extra fillers / artifacts
    "cause", "coz", "cos", "becau", "becuz",
    "ha", "haha", "hahaha", "yep", "nope", "duh", "ugh", "hmm",
    "lemme", "tryna", "imma", "outta", "kinda", "sorta",
    "thx", "thanks", "thank",
    "audience", "crowd", "stage", "mic", "show",
    "minute", "second",
    "told",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Audience-reaction markers: bracketed/parenthesized stage directions
# that aren't spoken by the comedian. Used both to clean text for analysis
# and to mine the n-grams that precede them (laughter triggers).
LAUGHTER_KEYWORDS = (
    "laughter", "laughs", "laughing", "chuckles", "chuckling", "giggles",
    "guffaws", "snickers",
)
APPLAUSE_KEYWORDS = (
    "applause", "cheers", "cheering", "clapping", "whistling",
)
_REACTION_INNER = "|".join(LAUGHTER_KEYWORDS + APPLAUSE_KEYWORDS)
REACTION_RE = re.compile(
    r"[\[\(](?:[^\]\)\n]{0,40}(?:" + _REACTION_INNER + r")[^\]\)\n]{0,40})[\]\)]",
    re.IGNORECASE,
)
LAUGHTER_RE = re.compile(
    r"[\[\(][^\]\)\n]{0,40}(?:" + "|".join(LAUGHTER_KEYWORDS) + r")[^\]\)\n]{0,40}[\]\)]",
    re.IGNORECASE,
)
APPLAUSE_RE = re.compile(
    r"[\[\(][^\]\)\n]{0,40}(?:" + "|".join(APPLAUSE_KEYWORDS) + r")[^\]\)\n]{0,40}[\]\)]",
    re.IGNORECASE,
)
# Any bracketed stage direction (we strip these for analysis too)
_ANY_STAGE_RE = re.compile(r"[\[\(][^\]\)\n]{0,60}[\]\)]")

# Common English contractions. Expanded so the tokenizer doesn't split
# "don't" into "don" + "t" and trigger spurious NRC matches for "don".
_CONTRACTIONS = {
    r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot", r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not", r"\bwouldn't\b": "would not",
    r"\bwon't\b": "will not", r"\bisn't\b": "is not", r"\baren't\b": "are not",
    r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not", r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not", r"\bmustn't\b": "must not",
    r"\bain't\b": "is not",
    r"\bi'm\b": "i am", r"\bi've\b": "i have", r"\bi'd\b": "i would",
    r"\bi'll\b": "i will",
    r"\byou're\b": "you are", r"\byou've\b": "you have",
    r"\byou'll\b": "you will", r"\byou'd\b": "you would",
    r"\bhe's\b": "he is", r"\bshe's\b": "she is", r"\bit's\b": "it is",
    r"\bwe're\b": "we are", r"\bwe've\b": "we have",
    r"\bwe'll\b": "we will", r"\bwe'd\b": "we would",
    r"\bthey're\b": "they are", r"\bthey've\b": "they have",
    r"\bthey'll\b": "they will", r"\bthey'd\b": "they would",
    r"\bthat's\b": "that is", r"\bthere's\b": "there is",
    r"\bwhat's\b": "what is", r"\bwhere's\b": "where is",
    r"\bwho's\b": "who is", r"\bhow's\b": "how is",
    r"\blet's\b": "let us",
    r"\bgonna\b": "going to", r"\bwanna\b": "want to", r"\bgotta\b": "got to",
}
_CONTRACTION_PATTERNS = [(re.compile(k, re.IGNORECASE), v) for k, v in _CONTRACTIONS.items()]


_HAHA_RE = re.compile(r"\b(?:ha\W*){2,}\b", re.IGNORECASE)
# A repeated single token, 3+ in a row, possibly with !/. between
# (e.g. "Louis! Louis! Louis!" chants before walking on stage).
_CHANT_RE = re.compile(
    r"\b([A-Za-z]{3,}!?)([\s!.,?]+)(?:\1\2){2,}\1?\b",
    re.IGNORECASE,
)


def clean_for_analysis(text: str) -> str:
    """
    Return a cleaned version of a transcript suitable for NLP analysis:
    - Normalize curly apostrophes (’ ` ´) to straight (') so the
      contraction patterns match.
    - Strip audience-reaction markers ([laughter], (applause), etc).
    - Strip any other bracketed stage direction ([man], [music]).
    - Strip "ha ha" / "haha" sequences (laughter approximations).
    - Strip audience chants of repeated names ("Louis! Louis! Louis!").
    - Expand common contractions so the tokenizer doesn't produce
      bogus tokens like "don" from "don't" or "re" from "you're".
    - Strip lyric music-note marks (♪).
    """
    if not isinstance(text, str):
        return ""
    # 1. Normalize apostrophes/quotes
    s = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    # 2. Strip stage directions
    s = _ANY_STAGE_RE.sub(" ", s)
    # 3. Strip "ha ha" / "haha" sequences
    s = _HAHA_RE.sub(" ", s)
    # 4. Strip repeated-token chants ("Louis! Louis! Louis!")
    s = _CHANT_RE.sub(" ", s)
    # 5. Expand contractions
    for pat, repl in _CONTRACTION_PATTERNS:
        s = pat.sub(repl, s)
    # 6. Music note marks used in lyric transcripts (e.g. Bo Burnham: Inside)
    s = s.replace("♪", " ")
    return re.sub(r"\s+", " ", s).strip()


_LIVE_AT_RE = re.compile(r"\s+(live|on|at)\s+.*$", re.IGNORECASE)
_TRAIL_YEAR_RE = re.compile(r"\s+\d{4}\s*$")
_EVENING_WITH_RE = re.compile(r"^an\s+evening\s+with\s+", re.IGNORECASE)
_POSSESSIVE_RE = re.compile(r"'s\b.*$", re.IGNORECASE)

# Manual overrides for titles where the scraping couldn't identify the comedian.
# Verified against the original title in the source.
COMEDIAN_OVERRIDES = {
    "The Age Of Spin": "Dave Chappelle",
    "Deep In The Heart Of Texas": "Dave Chappelle",
    "On Location": "George Carlin",
    "This Filthy World": "John Waters",
    "Demetri Martin. Person": "Demetri Martin",
}

# Titles that are not stand-up specials and should be dropped entirely.
DROP_TITLES = {"American Movie"}


def _comedian_from_title(t):
    """Best-effort: extract a comedian's name from a transcript title."""
    if not isinstance(t, str):
        return None
    base = t
    base = _EVENING_WITH_RE.sub("", base)
    # Split on any common separator and keep the first chunk
    base = re.split(r"\.\.\.|[:\-–&]", base, maxsplit=1)[0]
    # Strip possessive + everything after ("John Leguizamo's Road..." -> "John Leguizamo")
    base = _POSSESSIVE_RE.sub("", base)
    base = _TRAIL_YEAR_RE.sub("", base)
    base = _LIVE_AT_RE.sub("", base)
    base = base.strip(" .,!?")
    return base.strip().title() if base else None


def _canonicalize_comedians(names: pd.Series) -> dict:
    """
    Build a {variant -> canonical} map by collapsing variants that share
    a name prefix. Canonical = the shortest name in each group.
    """
    def key(s):
        if not isinstance(s, str):
            return ""
        # Drop everything except letters and spaces; periods inside names
        # ("C.K.") differ across variants ("Louis C.K." vs "Louis C.K"),
        # so we collapse them here to match both.
        s = re.sub(r"[^a-z ]", "", s.lower())
        toks = s.split()
        return " ".join(toks[:2])  # first two tokens identify the person

    groups = {}
    for n in names.dropna().unique():
        k = key(n)
        if not k:
            continue
        groups.setdefault(k, []).append(n)

    canon = {}
    for k, variants in groups.items():
        # Canonical = shortest token count, then prefer the form that
        # preserves periods (initialisms like "Louis C.K."), then shortest
        # length, then alphabetical for determinism.
        chosen = min(
            variants,
            key=lambda x: (len(x.split()), -x.count("."), len(x), x),
        )
        for v in variants:
            canon[v] = chosen
    return canon


def load_unified(path: str | Path = UNIFIED_PATH) -> pd.DataFrame:
    """Load df_unified.parquet and add derived columns (cleaned comedian, runtime, counts)."""
    df = pd.read_parquet(path).copy()

    mask_bad = (
        df["comedian"].isin(["Stand-up transcripts", None, ""]) | df["comedian"].isna()
    )
    df.loc[mask_bad, "comedian"] = df.loc[mask_bad, "title"].map(_comedian_from_title)

    # Some valid `comedian` values still contain the special name because
    # the source title had no clean separator. Re-parse those too.
    needs_clean = df["comedian"].map(
        lambda c: isinstance(c, str) and (
            bool(re.search(r"\d{4}", c))            # trailing year
            or bool(re.search(r"&|\.\.\.|:", c))     # leftover punctuation
            or bool(_EVENING_WITH_RE.match(c))       # "An Evening With ..."
        )
    ).fillna(False).astype(bool)
    df.loc[needs_clean, "comedian"] = df.loc[needs_clean, "comedian"].map(_comedian_from_title)

    # Normalize casing (e.g. "JUDAH FRIEDLANDER" -> "Judah Friedlander")
    df["comedian"] = df["comedian"].map(
        lambda c: c.title() if isinstance(c, str) and c.isupper() else c
    )

    # Manual overrides for titles whose comedian the scraper couldn't infer.
    df["comedian"] = df["comedian"].map(lambda c: COMEDIAN_OVERRIDES.get(c, c))

    # Drop titles that aren't stand-up specials (e.g. unrelated documentaries).
    df = df[~df["comedian"].isin(DROP_TITLES)].reset_index(drop=True)

    # Final pass: collapse variants of the same person (e.g. "Louis C.K. Oh My God")
    canon = _canonicalize_comedians(df["comedian"])
    df["comedian"] = df["comedian"].map(lambda c: canon.get(c, c))

    df["runtime_min"] = df["runtimes"].map(
        lambda x: int(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    df["transcript_raw"] = df["transcript"].fillna("")
    df["transcript"] = df["transcript_raw"].map(clean_for_analysis)
    df["word_count"] = df["transcript"].str.split().str.len()
    return df


def apply_filters(
    df: pd.DataFrame,
    comedians: list | None = None,
    year_range: tuple | None = None,
    min_rating: float = 0.0,
    min_votes: int = 0,
    top_n_by_votes: int | None = None,
    titles: list | None = None,
) -> pd.DataFrame:
    """Filter the unified dataframe. All params optional and orthogonal."""
    out = df.copy()
    if comedians:
        out = out[out["comedian"].isin(comedians)]
    if titles:
        out = out[out["title"].isin(titles)]
    if year_range:
        lo, hi = year_range
        out = out[(out["year"].between(lo, hi)) | out["year"].isna()]
    if min_rating:
        out = out[out["rating"].fillna(0) >= min_rating]
    if min_votes:
        out = out[out["votes"].fillna(0) >= min_votes]
    if top_n_by_votes:
        out = out.nlargest(top_n_by_votes, "votes")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tokenization + basic features
# ---------------------------------------------------------------------------
def tokenize(text: str, stopwords: set | None = None, min_len: int = 3) -> list[str]:
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in sw and len(t) >= min_len]


def top_ngrams(corpus: Iterable[str], n: int = 1, top_k: int = 20,
               stopwords: set | None = None) -> pd.Series:
    from sklearn.feature_extraction.text import CountVectorizer
    sw = list(stopwords if stopwords is not None else DEFAULT_STOPWORDS)
    vec = CountVectorizer(ngram_range=(n, n), stop_words=sw, min_df=2)
    X = vec.fit_transform(corpus)
    sums = X.sum(axis=0).A1
    return pd.Series(sums, index=vec.get_feature_names_out()).nlargest(top_k)


def sentiment_compound(text: str) -> float:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    if not hasattr(sentiment_compound, "_sia"):
        sentiment_compound._sia = SentimentIntensityAnalyzer()
    return sentiment_compound._sia.polarity_scores(text or "")["compound"]


# ---------------------------------------------------------------------------
# NRC emotion lexicon (8 emotions + 2 sentiments)
# ---------------------------------------------------------------------------
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def emotion_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame [show_idx, emotion, score] where score is the share
    of NRC-tagged words for that emotion. Uses the NRCLex package which
    bundles the NRC Word-Emotion Association Lexicon.
    """
    from nrclex import NRCLex

    rows = []
    for idx, row in df.iterrows():
        nrc = NRCLex()
        nrc.load_raw_text(row["transcript"] or "x")
        freq = nrc.affect_frequencies  # dict with emotions + 'positive'/'negative'
        for e in EMOTIONS:
            rows.append({
                "show_idx": idx,
                "comedian": row["comedian"],
                "title": row["title"],
                "rating": row.get("rating"),
                "emotion": e,
                "score": float(freq.get(e, 0.0)),
            })
    return pd.DataFrame(rows)


def emotion_top_words(df: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    For each emotion, return the top-K words in the subset that triggered
    it (NRC affect_dict maps each word to the emotions it expresses).
    Returns a long-form DataFrame [emotion, word, count, share].
    """
    from nrclex import NRCLex
    from collections import Counter

    counters: dict[str, Counter] = {e: Counter() for e in EMOTIONS}
    for _, row in df.iterrows():
        nrc = NRCLex()
        nrc.load_raw_text(row["transcript"] or "x")
        # affect_dict: {word: [emotion1, emotion2, ...]} per occurrence type;
        # raw_emotion_scores gives total counts per emotion. We re-tokenize
        # to know which raw word contributed to each emotion.
        words = nrc.words
        for w in words:
            wl = w.lower()
            emos = nrc.affect_dict.get(wl, [])
            for emo in emos:
                if emo in counters:
                    counters[emo][wl] += 1

    rows = []
    for emo, c in counters.items():
        total = sum(c.values())
        for w, n in c.most_common(top_k):
            rows.append({"emotion": emo, "word": w, "count": n,
                         "share": n / total if total else 0.0})
    return pd.DataFrame(rows)


def laughter_triggers(
    df: pd.DataFrame,
    kind: str = "laughter",
    n_words: int = 6,
    top_k: int = 25,
    ngram_min: int = 2,
    ngram_max: int = 4,
) -> dict:
    """
    Find what the comedian was saying *just before* the audience laughs or
    applauds. Uses the raw transcript (with reaction markers preserved).

    Args:
        kind: "laughter", "applause", or "both".
        n_words: how many tokens before each marker to consider.
        top_k: how many of the most frequent windows / n-grams to keep.
        ngram_min/max: n-gram range for the trigger phrase ranking.

    Returns dict with:
        - "total_markers": int, count of markers in the subset.
        - "windows": list[str], the raw n-word snippets before each marker.
        - "top_ngrams": Series, top frequent n-gram triggers (with
          stopwords filtered).
    """
    from sklearn.feature_extraction.text import CountVectorizer

    if kind == "laughter":
        rx = LAUGHTER_RE
    elif kind == "applause":
        rx = APPLAUSE_RE
    else:
        rx = REACTION_RE

    src = df["transcript_raw"] if "transcript_raw" in df.columns else df["transcript"]

    windows = []
    for text in src.fillna(""):
        # Tokenize ignoring stage directions
        for m in rx.finditer(text):
            before = text[:m.start()]
            # Strip other stage directions inside the window so we get spoken words
            before_clean = _ANY_STAGE_RE.sub(" ", before)
            words = re.findall(r"[A-Za-z']+", before_clean)
            if len(words) >= 2:
                window = " ".join(words[-n_words:]).lower()
                windows.append(window)

    result = {
        "total_markers": len(windows),
        "windows": windows[:top_k * 3],  # keep some for display
        "top_ngrams": pd.Series(dtype=float),
        "top_last_words": pd.Series(dtype=float),
    }
    if not windows:
        return result

    # Top n-grams across all windows
    vec = CountVectorizer(
        ngram_range=(ngram_min, ngram_max),
        stop_words=list(DEFAULT_STOPWORDS),
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    try:
        M = vec.fit_transform(windows)
        sums = M.sum(axis=0).A1
        ng = pd.Series(sums, index=vec.get_feature_names_out()).nlargest(top_k)
        result["top_ngrams"] = ng
    except ValueError:
        pass

    # Last word before each laugh — the most "trigger-y" single token
    last_words = []
    for w in windows:
        toks = [t for t in w.split() if t not in DEFAULT_STOPWORDS and len(t) >= 3]
        if toks:
            last_words.append(toks[-1])
    if last_words:
        result["top_last_words"] = pd.Series(last_words).value_counts().head(top_k)
    return result


def yearly_word_trends(
    df: pd.DataFrame,
    words: list[str] | None = None,
    bucket: int = 5,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    For a list of words (or, if None, the top-K most common words in the
    full subset), compute their share per year-bucket (default 5 years).
    Returns wide DataFrame [bucket, word1, word2, ...] with relative
    frequencies (per 1000 tokens) per bucket.
    """
    sub = df.dropna(subset=["year"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["bucket"] = (sub["year"].astype(int) // bucket) * bucket
    if words is None:
        # Use top-K most-common tokens across the entire subset
        all_tokens = []
        for t in sub["transcript"].fillna(""):
            all_tokens.extend(tokenize(t))
        words = [w for w, _ in pd.Series(all_tokens).value_counts().head(top_k).items()]

    rows = []
    for b, group in sub.groupby("bucket"):
        bag = []
        for t in group["transcript"].fillna(""):
            bag.extend(tokenize(t))
        total = max(len(bag), 1)
        counts = pd.Series(bag).value_counts()
        row = {"bucket": int(b), "_total_tokens": total, "_shows": len(group)}
        for w in words:
            row[w] = 1000 * counts.get(w, 0) / total
        rows.append(row)
    out = pd.DataFrame(rows).set_index("bucket").sort_index()
    return out


def yearly_emotion_trends(emo_df: pd.DataFrame, df: pd.DataFrame,
                          bucket: int = 5) -> pd.DataFrame:
    """Average emotion score per year-bucket. Returns wide df [bucket, emotion...]."""
    merged = emo_df.merge(
        df[["title", "year"]].drop_duplicates("title"),
        on="title", how="left",
    ).dropna(subset=["year"])
    merged["bucket"] = (merged["year"].astype(int) // bucket) * bucket
    out = merged.groupby(["bucket", "emotion"])["score"].mean().unstack()
    return out.sort_index()


def yearly_sentiment_rating(df: pd.DataFrame, bucket: int = 5) -> pd.DataFrame:
    """Average sentiment and rating per year-bucket."""
    sub = df.dropna(subset=["year"]).copy()
    sub["sentiment"] = sub["transcript"].map(sentiment_compound)
    sub["bucket"] = (sub["year"].astype(int) // bucket) * bucket
    return sub.groupby("bucket").agg(
        sentiment=("sentiment", "mean"),
        rating=("rating", "mean"),
        n_shows=("title", "count"),
    ).sort_index()


def emotion_summary(emo_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per emotion across the subset."""
    if emo_df.empty:
        return pd.DataFrame()
    return emo_df.groupby("emotion").agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        max_score=("score", "max"),
        argmax_show=("score", lambda s: emo_df.loc[s.idxmax(), "title"] if len(s) else None),
    ).sort_values("mean_score", ascending=False)


# ---------------------------------------------------------------------------
# Topic modeling (LDA)
# ---------------------------------------------------------------------------
def extract_topics(
    df: pd.DataFrame,
    n_topics: int = 8,
    n_top_words: int = 10,
    max_df: float = 0.85,
    min_df: int = 2,
    random_state: int = 42,
) -> dict:
    """
    Run LDA on the transcript corpus. Returns:
        {
          "topics": list of (topic_id, [words])  # top words per topic
          "doc_topic": ndarray shape (n_docs, n_topics)  # mixing weights
          "labels": list of one-word labels (top word of each topic)
        }
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        stop_words=list(DEFAULT_STOPWORDS),
        max_df=max_df, min_df=min_df, ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    X = vec.fit_transform(df["transcript"].fillna(""))
    vocab = vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
    )
    doc_topic = lda.fit_transform(X)

    topics = []
    labels = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_top_words]
        words = [vocab[j] for j in top_idx]
        topics.append((i, words))
        labels.append(words[0])
    return {"topics": topics, "doc_topic": doc_topic, "labels": labels}


# ---------------------------------------------------------------------------
# Catchphrases — n-grams distinctive of a comedian
# ---------------------------------------------------------------------------
def catchphrases_by_comedian(
    df: pd.DataFrame,
    ngram_range: tuple = (2, 4),
    top_k: int = 10,
    min_df: int = 2,
    background: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """
    Return the n-grams with highest TF-IDF per comedian — i.e. used often
    by this comedian but rarely by the rest.

    When the subset has a single comedian, `background` (typically the
    full unfiltered df) is used as the comparison corpus so distinctive
    n-grams can still be computed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    by_com = df.groupby("comedian")["transcript"].apply(lambda x: " ".join(x.fillna("")))

    # Single-comedian case: compare against the full background corpus.
    if len(by_com) < 2:
        if background is None or len(background) < 2:
            return {com: pd.Series(dtype=float) for com in by_com.index}
        target_com = by_com.index[0]
        bg = background[background["comedian"] != target_com]
        if len(bg) == 0:
            return {target_com: pd.Series(dtype=float)}
        # Aggregate background by comedian to keep doc count reasonable
        bg_by_com = bg.groupby("comedian")["transcript"].apply(lambda x: " ".join(x.fillna("")))
        corpus = pd.concat([by_com, bg_by_com])
        vec = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_df, max_df=0.95,
            stop_words=list(DEFAULT_STOPWORDS), sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        M = vec.fit_transform(corpus)
        vocab = vec.get_feature_names_out()
        scores = pd.Series(M[0].toarray().ravel(), index=vocab)
        return {target_com: scores.nlargest(top_k)}

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.95,
        stop_words=list(DEFAULT_STOPWORDS),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    M = vec.fit_transform(by_com)
    vocab = vec.get_feature_names_out()

    out = {}
    for i, com in enumerate(by_com.index):
        scores = pd.Series(M[i].toarray().ravel(), index=vocab)
        out[com] = scores.nlargest(top_k)
    return out


# ---------------------------------------------------------------------------
# Clustering with UMAP/t-SNE
# ---------------------------------------------------------------------------
def cluster_comedians(
    df: pd.DataFrame,
    k: int = 5,
    method: str = "umap",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster comedians by TF-IDF style and project to 2D for visualization.
    Returns DataFrame [comedian, cluster, x, y, n_shows].
    """
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    by_com = df.groupby("comedian").agg(
        transcript=("transcript", lambda x: " ".join(x.fillna(""))),
        n_shows=("title", "count"),
    )
    if len(by_com) < max(k, 3):
        return pd.DataFrame(columns=["comedian", "cluster", "x", "y", "n_shows"])

    vec = TfidfVectorizer(
        stop_words=list(DEFAULT_STOPWORDS), max_df=0.9, min_df=1, ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    M = vec.fit_transform(by_com["transcript"])

    k_eff = min(k, len(by_com))
    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    clusters = km.fit_predict(M)

    n_samples = M.shape[0]
    if method == "umap":
        try:
            import umap
            n_neighbors = max(2, min(15, n_samples - 1))
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors)
            coords = reducer.fit_transform(M.toarray())
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = max(2.0, min(30.0, (n_samples - 1) / 3))
        coords = TSNE(n_components=2, random_state=random_state, perplexity=perplexity).fit_transform(M.toarray())

    return pd.DataFrame({
        "comedian": by_com.index,
        "cluster": clusters,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "n_shows": by_com["n_shows"].values,
    })


# ---------------------------------------------------------------------------
# Predictive model of the rating
# ---------------------------------------------------------------------------
def predict_rating(df: pd.DataFrame, alpha: float = 1.0, max_features: int = 2000) -> dict:
    """
    Fit a Ridge regression on TF-IDF + numeric features to predict IMDb rating.
    Returns:
        {
          "score_train": R^2 on training,
          "score_cv":    mean R^2 in 5-fold CV (if n>=10),
          "n":           sample size,
          "top_positive": Series of features that push rating UP,
          "top_negative": Series of features that push rating DOWN,
        }
    Built for interpretability, not predictive power (n is small).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from scipy.sparse import hstack, csr_matrix

    sub = df.dropna(subset=["rating", "transcript"]).copy()
    sub = sub[sub["transcript"].str.len() > 100]
    if len(sub) < 5:
        return {"error": f"Solo {len(sub)} shows con rating y transcript — insuficiente."}

    vec = TfidfVectorizer(
        stop_words=list(DEFAULT_STOPWORDS),
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    X_text = vec.fit_transform(sub["transcript"])
    feature_names = list(vec.get_feature_names_out())

    extra = sub[["word_count", "runtime_min"]].fillna(sub[["word_count", "runtime_min"]].median())
    extra_norm = (extra - extra.mean()) / (extra.std().replace(0, 1))
    X_extra = csr_matrix(extra_norm.values)
    feature_names += [f"num::{c}" for c in extra.columns]

    X = hstack([X_text, X_extra]).tocsr()
    y = sub["rating"].values

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    score_train = model.score(X, y)

    score_cv = None
    if len(sub) >= 10:
        cv_k = min(5, len(sub))
        scores = cross_val_score(Ridge(alpha=alpha), X, y, cv=cv_k, scoring="r2")
        score_cv = float(scores.mean())

    coefs = pd.Series(model.coef_, index=feature_names).sort_values()
    return {
        "n": int(len(sub)),
        "score_train": float(score_train),
        "score_cv": score_cv,
        "top_negative": coefs.head(15),
        "top_positive": coefs.tail(15)[::-1],
    }


# ---------------------------------------------------------------------------
# Narratives — auto-generated text descriptions of each result
# ---------------------------------------------------------------------------
def narrate_filters(df_all: pd.DataFrame, df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "**Filtros vacíos:** ningún show coincide. Relaja los filtros."
    pct = 100 * len(df) / max(len(df_all), 1)
    n_com = df["comedian"].nunique()
    yrs = df["year"].dropna()
    rng = f"{int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "n/d"
    rt = df["rating"].dropna()
    rt_str = f"{rt.mean():.2f} (n={len(rt)})" if len(rt) else "sin ratings"
    return (
        f"**Subset actual:** {len(df)} shows ({pct:.0f}% del total), "
        f"{n_com} comediantes, años {rng}. Rating IMDb medio: **{rt_str}**."
    )


def narrate_topics(result: dict, df: pd.DataFrame) -> str:
    topics = result["topics"]
    doc_topic = result["doc_topic"]
    dominant = doc_topic.argmax(axis=1)
    counts = Counter(dominant)
    top_topic, top_count = counts.most_common(1)[0]
    top_words = ", ".join(topics[top_topic][1][:5])
    n_topics = len(topics)
    return (
        f"Se detectaron **{n_topics} tópicos** latentes. El dominante "
        f"({top_count} shows, {100 * top_count / len(dominant):.0f}%) gira "
        f"alrededor de: *{top_words}*. Cada show es una mezcla de los "
        f"{n_topics} tópicos en distintas proporciones."
    )


def narrate_emotions(emo_df: pd.DataFrame) -> str:
    if emo_df.empty:
        return "Sin datos de emociones."
    avg = emo_df.groupby("emotion")["score"].mean().sort_values(ascending=False)
    top = avg.head(2)
    bot = avg.tail(1)
    parts = ", ".join(f"**{e}** ({s:.1%})" for e, s in top.items())
    return (
        f"Emociones dominantes (NRC) en el subset: {parts}. "
        f"La menos presente: **{bot.index[0]}** ({bot.iloc[0]:.1%})."
    )


def narrate_catchphrases(cp: dict[str, pd.Series], top_n: int = 3) -> str:
    if not cp:
        return "Sin catchphrases para mostrar."
    lines = ["**Frases distintivas por comediante** (TF-IDF sobre n-gramas):"]
    for com, ser in list(cp.items())[:top_n]:
        if len(ser):
            phrases = " · ".join(f"*{p}*" for p in ser.index[:5])
            lines.append(f"- **{com}**: {phrases}")
    return "\n".join(lines)


def narrate_predictor(result: dict) -> str:
    if "error" in result:
        return f"⚠️ {result['error']}"
    pos = ", ".join(f"*{w.replace('num::', '')}*" for w in result["top_positive"].head(5).index)
    neg = ", ".join(f"*{w.replace('num::', '')}*" for w in result["top_negative"].head(5).index)
    cv_part = f" · R² CV-5: **{result['score_cv']:.2f}**" if result["score_cv"] is not None else ""
    return (
        f"Modelo Ridge (n={result['n']}). R² train: **{result['score_train']:.2f}**{cv_part}. "
        f"Features que **suben** el rating: {pos}. Las que lo **bajan**: {neg}. "
        f"⚠️ Con n pequeño y palabras como features, los coeficientes son sugerentes, no causales."
    )
