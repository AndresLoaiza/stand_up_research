import re
from pathlib import Path

from bs4 import BeautifulSoup
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FRAME_DIR = PROJECT_ROOT / "data" / "data_frame"


def extract_transcript_urls() -> list:
    """Scrape https://scrapsfromtheloft.com/stand-up-comedy-scripts/ for all transcript URLs."""
    url = "https://scrapsfromtheloft.com/stand-up-comedy-scripts/"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, "lxml")
    cards_shows = soup.find_all("div", class_="elementor-post__text")
    return [card.find("a").get("href").strip("/") for card in cards_shows]


def extract_transcript(stand_up_list: list) -> pd.DataFrame:
    """For each transcript URL, scrape title, comedian and full text."""
    rows = []
    for stand_up_url in stand_up_list:
        html_text = requests.get(stand_up_url).text
        soup = BeautifulSoup(html_text, "lxml")
        text_body = soup.find_all(
            "div",
            class_="elementor-element elementor-element-74af9a5b elementor-widget "
                   "elementor-widget-theme-post-content",
        )
        if not text_body:
            continue
        comedian = soup.find_all("a", class_="elementor-post-info__terms-list-item")[1].text
        title = soup.find("h1", class_="elementor-heading-title elementor-size-default").text
        strong_text = text_body[0].find_all("strong")
        paragraphs = text_body[0].find_all("p")
        transcript = "\n".join(p.text for p in paragraphs if p not in strong_text)
        rows.append(dict(
            title=re.sub(r"\s\(.*", "", title).upper(),
            comedian=comedian,
            transcript=transcript,
        ))
    return pd.DataFrame(rows)


def load_transcript_table() -> pd.DataFrame:
    return pd.read_parquet(DATA_FRAME_DIR / "raw_transcripts.parquet")


def save_transcript_table(df_transcript: pd.DataFrame) -> None:
    df_transcript.to_parquet(DATA_FRAME_DIR / "raw_transcripts.parquet")


def main():
    urls = extract_transcript_urls()
    df = extract_transcript(urls)
    save_transcript_table(df)


if __name__ == "__main__":
    main()
