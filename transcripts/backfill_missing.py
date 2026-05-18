"""
Backfill missing transcripts:
1. Scrape the full index page of scrapsfromtheloft.com (470+ URLs).
2. Match each show without transcript to a candidate URL using slug
   variants and a known-overrides table.
3. Fetch matched transcripts and update the parquets.
4. Drop shows we couldn't match (so the dataset only contains shows
   that actually have text).

Run from project root:
    python transcripts/backfill_missing.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DF_DIR = ROOT / "data" / "data_frame"
HEADERS = {"User-Agent": "Mozilla/5.0"}
INDEX_URL = "https://scrapsfromtheloft.com/stand-up-comedy-scripts/"
BODY_CLASS = (
    "elementor-element elementor-element-74af9a5b elementor-widget "
    "elementor-widget-theme-post-content"
)

# Known direct URL overrides (cases the slug matcher misses).
KNOWN_URLS = {
    "LOUIS C.K. OH MY GOD": "https://scrapsfromtheloft.com/comedy/louis-ck-oh-my-god-full-transcript/",
    "LOUIS C.K.: LIVE AT THE COMEDY STORE": "https://scrapsfromtheloft.com/comedy/louis-c-k-live-at-the-comedy-store-2015-transcript/",
    "LOUIS C.K. 2017": "https://scrapsfromtheloft.com/comedy/louis-c-k-2017-transcript/",
    "JOHN LEGUIZAMO'S ROAD TO BROADWAY": "https://scrapsfromtheloft.com/comedy/latin-history-for-morons-john-leguizamo-transcript/",
}


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"['’`´]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def slugify_compact_digits(text: str) -> str:
    """Same as slugify but joins adjacent digit groups: '8:46' -> '846'."""
    s = text.lower()
    s = re.sub(r"['’`´]", "", s)
    s = re.sub(r"(\d)[^a-z0-9]+(\d)", r"\1\2", s)  # 8:46 -> 846
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def normalize_slug(slug: str) -> str:
    """Collapse 'louis-c-k' to 'louis-ck' and strip leading 'the-'."""
    slug = re.sub(r"\b([a-z])-([a-z])\b", r"\1\2", slug)
    slug = re.sub(r"^the-", "", slug)
    return slug


def url_to_slug(url: str) -> str:
    slug = url.rsplit("/", 1)[-1]
    slug = re.sub(r"-(full-)?transcript$", "", slug)
    # Strip trailing year like '-2015' or '-2010'
    slug = re.sub(r"-(19|20)\d{2}$", "", slug)
    return slug


def fetch_index_urls() -> list[str]:
    r = requests.get(INDEX_URL, headers=HEADERS, verify=False, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if "/comedy/" in h and "transcript" in h:
            urls.add(h.rstrip("/"))
    return sorted(urls)


def best_match(title: str, comedian: str, url_index: dict[str, str]) -> str | None:
    """Find the URL whose slug best matches the title/comedian."""
    if not isinstance(title, str) or not title.strip():
        return None
    if title in KNOWN_URLS:
        return KNOWN_URLS[title]

    title_slugs = {
        slugify(title),
        slugify_compact_digits(title),
        normalize_slug(slugify(title)),
        normalize_slug(slugify_compact_digits(title)),
    }

    # 1) Exact slug match against any variant
    for slug, url in url_index.items():
        sn = normalize_slug(slug)
        if slug in title_slugs or sn in title_slugs:
            return url

    # 2) Containment (either direction, normalized)
    for slug, url in url_index.items():
        sn = normalize_slug(slug)
        for ts in title_slugs:
            if (ts in sn or sn in ts) and min(len(ts), len(sn)) >= 8:
                return url

    # 3) Comedian + a key token from the title tail
    if isinstance(comedian, str) and comedian.strip():
        com_norm = normalize_slug(slugify(comedian))
        tail = title.split(":", 1)[1] if ":" in title else title
        key_tokens = [t for t in slugify(tail).split("-") if len(t) >= 4]
        for slug, url in url_index.items():
            sn = normalize_slug(slug)
            if com_norm in sn and any(tok in sn for tok in key_tokens):
                return url
    return None


def fetch_transcript(url: str) -> str:
    r = requests.get(url, headers=HEADERS, verify=False, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    body = soup.find_all("div", class_=BODY_CLASS)
    if not body:
        raise RuntimeError("body not found")
    strong = {id(s) for s in body[0].find_all("strong")}
    paragraphs = body[0].find_all("p")
    lines = [p.text for p in paragraphs if id(p) not in strong]
    text = "\n".join(lines).strip()
    if len(text) < 500:
        raise RuntimeError(f"transcript too short ({len(text)} chars)")
    return text


def main():
    print("Loading parquets...")
    raw = pd.read_parquet(DF_DIR / "raw_transcripts.parquet")
    imdb = pd.read_parquet(DF_DIR / "df_imdb.parquet")
    uni = pd.read_parquet(DF_DIR / "df_unified.parquet")

    missing_mask = uni["transcript"].isna() | (uni["transcript"].fillna("").str.len() < 200)
    missing = uni[missing_mask].copy()
    print(f"Missing transcripts: {len(missing)} of {len(uni)}")

    print(f"Fetching index from {INDEX_URL}...")
    urls = fetch_index_urls()
    print(f"Index has {len(urls)} transcript URLs.")
    url_index = {url_to_slug(u): u for u in urls}

    matched, unmatched = [], []
    for idx, row in missing.iterrows():
        u = best_match(row["title"], row.get("comedian") or "", url_index)
        if u:
            matched.append((idx, row["title"], row.get("comedian") or "", u))
        else:
            unmatched.append((idx, row["title"], row.get("comedian") or ""))
    print(f"\nMatched: {len(matched)} | Unmatched: {len(unmatched)}")

    fetched, failed = [], []
    for i, (idx, title, comedian, url) in enumerate(matched, 1):
        print(f"  [{i:>3}/{len(matched)}] {title[:55]:55s}", end=" ... ")
        try:
            text = fetch_transcript(url)
            fetched.append((idx, title, comedian, url, text))
            print(f"OK ({len(text):>6} chars)")
        except Exception as e:
            failed.append((idx, title, comedian, url, str(e)))
            print(f"FAIL: {e}")
        time.sleep(0.3)

    print(f"\nFetched OK: {len(fetched)} | Failed: {len(failed)}")

    # Apply updates to df_unified
    drop_idx = [idx for idx, _, _ in unmatched] + [idx for idx, _, _, _, _ in failed]
    for idx, _, _, _, text in fetched:
        uni.at[idx, "transcript"] = text
    uni_new = uni.drop(index=drop_idx).reset_index(drop=True)

    # Append new transcripts to raw_transcripts
    raw_add = pd.DataFrame([
        {"title": t, "comedian": c, "transcript": text}
        for _, t, c, _, text in fetched
    ])
    raw_new = pd.concat([raw, raw_add], ignore_index=True) if len(raw_add) else raw

    # Drop corresponding rows from df_imdb (case-insensitive title match)
    surviving_titles_lower = {t.lower() for t in uni_new["title"]}
    imdb_new = imdb[imdb["title"].str.lower().isin(surviving_titles_lower)].reset_index(drop=True)

    raw_new.to_parquet(DF_DIR / "raw_transcripts.parquet")
    imdb_new.to_parquet(DF_DIR / "df_imdb.parquet")
    uni_new.to_parquet(DF_DIR / "df_unified.parquet")

    print()
    print(f"raw_transcripts: {len(raw)} -> {len(raw_new)} ({len(raw_new)-len(raw):+d})")
    print(f"df_imdb:         {len(imdb)} -> {len(imdb_new)} ({len(imdb_new)-len(imdb):+d})")
    print(f"df_unified:      {len(uni)} -> {len(uni_new)} ({len(uni_new)-len(uni):+d})")

    if unmatched:
        print(f"\nUnmatched ({len(unmatched)}, deleted) — sample:")
        for _, t, c in unmatched[:15]:
            print(f"  {(c or '?')[:25]:25s}  {t}")


if __name__ == "__main__":
    main()
