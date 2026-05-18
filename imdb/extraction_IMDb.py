from pathlib import Path

from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from imdb import IMDb

ia = IMDb()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIST_ID_DIR = PROJECT_ROOT / "data" / "list_id"
DATA_FRAME_DIR = PROJECT_ROOT / "data" / "data_frame"


def extract_imdb_id_movies():
    """
    Search IMDb with filters: type=tv_special|documentary, user_rating>=7.5,
    num_votes>=1000, genre=comedy, language=en, newest first. Returns the list
    of imdb ids in the result page.
    """
    url = (
        "https://www.imdb.com/search/title/?title_type=tv_special,documentary"
        "&user_rating=7.5,&num_votes=1000,&genres=comedy&languages=en"
        "&count=250&sort=release_date,desc"
    )
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, "lxml")
    time.sleep(1)
    cards_shows = soup.find_all("div", class_="lister-item-content")
    return [card.find("a").get("href")[9:].strip("/") for card in cards_shows]


def _read_id_list(filename: str) -> list:
    path = LIST_ID_DIR / filename
    with open(path, "r") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def _write_id_list(filename: str, ids: list) -> None:
    path = LIST_ID_DIR / filename
    with open(path, "w") as fh:
        for item in ids:
            fh.write(f"{item}\n")


def extract_imdb_id_to_delete() -> list:
    """IDs from the search that are not stand-up specials (manual curation)."""
    return _read_id_list("imdb_id_to_delete.txt")


def extract_imdb_id_to_add() -> list:
    """IDs missing from the search but that ARE stand-up specials (manual curation)."""
    return _read_id_list("imdb_id_to_add.txt")


def clean_imdb_id_list() -> list:
    """Combine the raw search with the manual add/delete curation."""
    raw_ids = extract_imdb_id_movies()
    to_delete = set(extract_imdb_id_to_delete())
    to_add = extract_imdb_id_to_add()
    clean = [i for i in raw_ids if i not in to_delete]
    clean.extend(i for i in to_add if i not in clean)
    return clean


def insert_id_to_delete(imdb_id_to_delete: list) -> None:
    _write_id_list("imdb_id_to_delete.txt", imdb_id_to_delete)


def insert_id_to_add(imdb_id_to_add: list) -> None:
    _write_id_list("imdb_id_to_add.txt", imdb_id_to_add)


def get_imdb_info(clean_imdb_id: list) -> list:
    """Fetch the raw IMDb Movie object for each id."""
    list_imdb = []
    for id_ in clean_imdb_id:
        list_imdb.append([id_, ia.get_movie(id_)])
        print(f"Loaded id: {id_}")
    return list_imdb


def extraction_imdb_features(list_imdb: list) -> pd.DataFrame:
    """Project the raw IMDb objects into a flat DataFrame."""
    rows = []
    for imdb_id, movie in list_imdb:
        def g(key, transform=lambda x: x):
            try:
                return transform(movie[key])
            except (KeyError, IndexError, TypeError):
                return None

        rows.append(dict(
            imdbID=imdb_id,
            title=g("title", lambda x: x.upper()),
            distributors=g("distributors", lambda x: x[0]["name"]),
            year=g("year"),
            plot=g("plot"),
            votes=g("votes"),
            original_title=g("original title"),
            writer=g("writer", lambda x: x[0]["name"]),
            runtimes=g("runtimes"),
            countries=g("countries"),
            original_air_date=g("original air date"),
            rating=g("rating"),
            director=g("director", lambda x: x[0]["name"]),
            demographics=ia.get_movie_vote_details(imdb_id)["data"]["demographics"],
        ))
    return pd.DataFrame.from_dict(rows)


def save_imdb_table(df_imdb: pd.DataFrame) -> None:
    df_imdb.to_parquet(DATA_FRAME_DIR / "df_imdb.parquet")


def load_imdb_table() -> pd.DataFrame:
    return pd.read_parquet(DATA_FRAME_DIR / "df_imdb.parquet")


def main():
    clean_ids = clean_imdb_id_list()
    raw_info = get_imdb_info(clean_ids)
    df_imdb = extraction_imdb_features(raw_info)
    save_imdb_table(df_imdb)


if __name__ == "__main__":
    main()
