"""
One-shot script to fetch transcripts + minimal IMDb metadata for the
8 post-2020 specials with rating >= 7.5, and merge them into the
existing parquet files.

Run from project root:
    python transcripts/fetch_new_specials.py
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

# Curated catalog: scrapsfromtheloft URL + verified IMDb metadata.
NEW_SPECIALS = [
    {
        "url": "https://scrapsfromtheloft.com/comedy/bo-burnham-inside-transcript/",
        "imdbID": "14544192", "comedian": "Bo Burnham",
        "title_clean": "BO BURNHAM: INSIDE",
        "year": 2021, "rating": 8.6, "runtime": 87,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/sincerely-louis-ck-transcript/",
        "imdbID": "12087624", "comedian": "Louis C.K.",
        "title_clean": "SINCERELY LOUIS C.K.",
        "year": 2020, "rating": 8.2, "runtime": 60,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/jerrod-carmichael-rothaniel-transcript/",
        "imdbID": "18949702", "comedian": "Jerrod Carmichael",
        "title_clean": "JERROD CARMICHAEL: ROTHANIEL",
        "year": 2022, "rating": 7.8, "runtime": 56,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/louis-ck-at-the-dolby-transcript/",
        "imdbID": "27430909", "comedian": "Louis C.K.",
        "title_clean": "LOUIS C.K. AT THE DOLBY",
        "year": 2023, "rating": 7.7, "runtime": 73,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/louis-c-k-sorry-transcript/",
        "imdbID": "16478030", "comedian": "Louis C.K.",
        "title_clean": "LOUIS C.K.: SORRY",
        "year": 2021, "rating": 7.7, "runtime": 64,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/norm-macdonald-nothing-special-transcript/",
        "imdbID": "20201450", "comedian": "Norm Macdonald",
        "title_clean": "NORM MACDONALD: NOTHING SPECIAL",
        "year": 2022, "rating": 7.5, "runtime": 64,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/mike-birbiglia-old-man-and-pool-transcript/",
        "imdbID": "29729075", "comedian": "Mike Birbiglia",
        "title_clean": "MIKE BIRBIGLIA: THE OLD MAN AND THE POOL",
        "year": 2023, "rating": 7.5, "runtime": 75,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/john-mulaney-baby-j-transcript/",
        "imdbID": "27141610", "comedian": "John Mulaney",
        "title_clean": "JOHN MULANEY: BABY J",
        "year": 2023, "rating": 7.5, "runtime": 80,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/dave-chappelle-the-dreamer-transcript/",
        "imdbID": "18278698", "comedian": "Dave Chappelle",
        "title_clean": "DAVE CHAPPELLE: THE DREAMER",
        "year": 2023, "rating": 7.0, "runtime": 58,
    },
    {
        "url": "https://scrapsfromtheloft.com/comedy/dave-chappelle-the-closer-transcript/",
        "imdbID": "15523010", "comedian": "Dave Chappelle",
        "title_clean": "DAVE CHAPPELLE: THE CLOSER",
        "year": 2021, "rating": 7.9, "runtime": 72,
    },
]

LEGACY_BODY_CLASS = (
    "elementor-element elementor-element-74af9a5b elementor-widget "
    "elementor-widget-theme-post-content"
)


def fetch_transcript(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=False, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    body = soup.find_all("div", class_=LEGACY_BODY_CLASS)
    if not body:
        raise RuntimeError(f"Cannot find transcript body in {url}")
    strong = set(id(s) for s in body[0].find_all("strong"))
    paragraphs = body[0].find_all("p")
    lines = [p.text for p in paragraphs if id(p) not in strong]
    return "\n".join(lines).strip()


def main():
    print(f"Loading existing parquets from {DF_DIR}...")
    raw = pd.read_parquet(DF_DIR / "raw_transcripts.parquet")
    imdb = pd.read_parquet(DF_DIR / "df_imdb.parquet")
    uni = pd.read_parquet(DF_DIR / "df_unified.parquet")

    existing_titles = set(uni["title"].astype(str))
    new_rows_raw, new_rows_imdb = [], []
    for spec in NEW_SPECIALS:
        if spec["title_clean"] in existing_titles:
            print(f"  skip (already present): {spec['title_clean']}")
            continue
        print(f"  fetching {spec['title_clean']}...")
        try:
            transcript = fetch_transcript(spec["url"])
        except Exception as e:
            print(f"    !! failed: {e}")
            continue
        if len(transcript) < 500:
            print(f"    !! transcript too short ({len(transcript)} chars), skipping")
            continue

        new_rows_raw.append({
            "title": spec["title_clean"],
            "comedian": spec["comedian"],
            "transcript": transcript,
        })
        new_rows_imdb.append({
            "imdbID": spec["imdbID"],
            "title": spec["title_clean"],
            "distributors": None,
            "year": spec["year"],
            "plot": None,
            "votes": None,
            "original_title": spec["title_clean"],
            "writer": None,
            "runtimes": [str(spec["runtime"])],
            "countries": None,
            "original_air_date": None,
            "rating": spec["rating"],
            "director": None,
            "demographics": None,
        })
        time.sleep(1)  # be polite

    if not new_rows_raw:
        print("Nothing new to add.")
        return

    raw_new = pd.concat([raw, pd.DataFrame(new_rows_raw)], ignore_index=True)
    imdb_new = pd.concat([imdb, pd.DataFrame(new_rows_imdb)], ignore_index=True)

    uni_add = pd.DataFrame(new_rows_imdb).copy()
    uni_add["comedian"] = [r["comedian"] for r in new_rows_raw]
    uni_add["transcript"] = [r["transcript"] for r in new_rows_raw]
    uni_new = pd.concat([uni, uni_add[uni.columns]], ignore_index=True)

    raw_new.to_parquet(DF_DIR / "raw_transcripts.parquet")
    imdb_new.to_parquet(DF_DIR / "df_imdb.parquet")
    uni_new.to_parquet(DF_DIR / "df_unified.parquet")

    print(f"\n+ {len(new_rows_raw)} specials added")
    print(f"  raw_transcripts: {len(raw)} -> {len(raw_new)}")
    print(f"  df_imdb:         {len(imdb)} -> {len(imdb_new)}")
    print(f"  df_unified:      {len(uni)} -> {len(uni_new)}")


if __name__ == "__main__":
    main()
