    # %%
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from imdb import IMDb
from imdb import helpers

# create an instance of the IMDb class
ia = IMDb()


def extract_imdb_id_movies():
    url = 'https://www.imdb.com/search/title/?title_type=tv_special,documentary&user_rating=7.5,&num_votes=1000,&genres=comedy&languages=en&count=250&sort=release_date,desc'
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')
    time.sleep(1)  # For some reason we have to wait
    cards_shows = soup.find_all('div', class_='lister-item-content')
    imdb_id = [stand_up.find('a').get('href')[9:].strip('/') for stand_up in cards_shows]
    return imdb_id


def extract_imdb_id_to_delete():
    # define an empty list
    imdb_id_to_delete = []

    # open file and read the content in a list
    with open('imdb_id_to_delete.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            id = line[:-1]

            # add item to the list
            imdb_id_to_delete.append(id)
    return imdb_id_to_delete


def extract_imdb_id_to_add():
    # define an empty list
    imdb_id_to_add = []

    # open file and read the content in a list
    with open('imdb_id_to_add.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            id = line[:-1]

            # add item to the list
            imdb_id_to_add.append(id)
    return imdb_id_to_add


def clean_imdb_id_list():
    raw_imdb_id = extract_imdb_id_movies()
    clean_imdb_id = [id for id in raw_imdb_id if id not in extract_imdb_id_to_delete()]
    clean_imdb_id.append(id for id in extract_imdb_id_to_add())
    return clean_imdb_id[:len(clean_imdb_id) - 1]


def insert_id_to_delete(imdb_id_to_delete: list):
    # escribir imdb_id_to_delete
    with open('imdb_id_to_delete.txt', 'w') as filehandle:
        for listitem in imdb_id_to_delete:
            filehandle.write('%s\n' % listitem)


def insert_id_to_add(imdb_id_to_add: list):
    # escribir imdb_id_to_add
    with open('imdb_id_to_add.txt', 'w') as filehandle:
        for listitem in imdb_id_to_add:
            filehandle.write('%s\n' % listitem)


def get_imdb_info(clean_imdb_id: list):
    list_imdb = []
    for id in clean_imdb_id:
        list_imdb.append([id, ia.get_movie(id)])
        print(f'Se cargó correctamente el id: {id}')

    return list_imdb

    # %%


def extraction_imdb_features(list_imdb):
    list_dict_movie_detail = []
    for imdb_movie in list_imdb:
        # distributors,genres,imdbID,kind,languages,original air date,plot,votes
        title = imdb_movie[1]['title']
        try:
            distributors = imdb_movie[1]['distributors'][0]['name']
        except:
            distributors = None
        year = imdb_movie[1]['year']
        plot = imdb_movie[1]['plot']
        votes = imdb_movie[1]['votes']
        original_title = imdb_movie[1]['original title']
        try:
            writer = imdb_movie[1]['writer'][0]['name']
        except:
            writer = None
        runtimes = imdb_movie[1]['runtimes']
        try:
            countries = imdb_movie[1]['countries']
        except:
            countries = None
        try:
            original_air_date = imdb_movie[1]['original air date']
        except:
            original_air_date = None
        rating = imdb_movie[1]['rating']
        try:
            director = imdb_movie[1]['director'][0]['name']
        except:
            director = None

        list_dict_movie_detail.append(dict(
            imdbID=imdb_movie[0],
            # imdb_movie = imdb_movie[1],
            title=title,
            distributors=distributors,
            year=year,
            plot=plot,
            votes=votes,
            original_title=original_title,
            writer=writer,
            runtimes=runtimes,
            countries=countries,
            original_air_date=original_air_date,
            rating=rating,
            director=director,
            demographics=ia.get_movie_vote_details(imdb_movie[0])['data']['demographics']
        ))

    return pd.DataFrame.from_dict(list_dict_movie_detail)
    # https://scrapsfromtheloft.com/stand-up-comedy-scripts/
    # %%


def save_imdb_table(df_imdb):
    df_imdb.to_parquet('df_imdb.parquet')
    # %%


def load_imdb_table():
    return pd.read_parquet('df_imdb.parquet')
    # %%
    #############################################
    ###
    ### TODO:
    ### 1.  Create feature to have unique codes
    ###     to the files (delete and add)
    #############################################


def _main_():
    # %%
    clean_imdb_ids_list = clean_imdb_id_list()
    list_raw_info = get_imdb_info(clean_imdb_ids_list)
    df_imdb = extraction_imdb_features(list_raw_info)
    # %%
    save_imdb_table(df_imdb)
    # %%
    df_test = load_imdb_table()
    # %%
    df_test
