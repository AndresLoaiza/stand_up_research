    # %%
from imdb import IMDb
from imdb import helpers
# create an instance of the IMDb class
ia = IMDb()
    # %%
# get a movie and print its director(s)
paper_tiger = ia.get_movie('11318624')
    # %%
for director in paper_tiger['directors']:
    print(director['name'])
    # %%
# show all information that are currently available for a movie
print(sorted(paper_tiger.keys()))
    # %%
# show all information sets that can be fetched for a movie
print(ia.get_movie_infoset())
    # %%
# update a Movie object with more information
ia.update(paper_tiger, ['technical'])
    # %%
# show which keys were added by the information set
print(paper_tiger.infoset2keys['technical'])
    # %%
# print one of the new keys
print(paper_tiger.get('tech'))
keys
['akas', 'aspect ratio', 'camera department', 'canonical title', 'cast', 'certificates', 'color info', 'countries',
 'country codes', 'cover url', 'director', 'directors', 'distributors', 'editorial department', 'editors',
 'full-size cover url', 'genres', 'imdbID', 'kind', 'language codes', 'languages', 'long imdb canonical title',
 'long imdb title', 'original air date', 'original title', 'plot', 'producers', 'production companies',
 'production managers', 'rating', 'runtimes', 'smart canonical title', 'smart long imdb canonical title', 'sound mix',
 'tech', 'title', 'votes', 'writer', 'writers', 'year']

    # %%
#distributors,genres,imdbID,kind,languages,original air date,plot,votes
paper_tiger['votes']

    # %%
paper_tiger['year']
    # %%
    ia.get_movie_vote_details('11318624')

    # %%
ia.get_movie_vote_details('10847306')['data']['demographics']


    # %%

a = helpers.get_byURL('https://www.imdb.com/title/tt11620828/?ref_=adv_li_tt')
type(a)


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
    soup = BeautifulSoup(html_text,'lxml')
    time.sleep(1) #For some reason we have to wait
    cards_shows = soup.find_all('div', class_ = 'lister-item-content' )
    imdb_id = [stand_up.find('a').get('href')[9:].strip('/') for stand_up in cards_shows]
    return imdb_id

    # %%

    # define an empty list
    imdb_id_to_delete = []

    # open file and read the content in a list
    with open('imdb_id_to_delete.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            id = line[:-1]

            # add item to the list
            imdb_id_to_delete.append(id)
    # %%

    # define an empty list
    imdb_id_to_add = []

    # open file and read the content in a list
    with open('imdb_id_to_add.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            id = line[:-1]

            # add item to the list
            imdb_id_to_add.append(id)
    # %%
    raw_imdb_id = extract_imdb_id_movies()
    clean_imdb_id = [id for id in raw_imdb_id if id not in imdb_id_to_delete ]
    clean_imdb_id.append(id for id in imdb_id_to_add)
    clean_imdb_id = clean_imdb_id[:len(clean_imdb_id) - 1]
    # %%


    #escribir imdb_id_to_delete
    with open('imdb_id_to_delete.txt', 'w') as filehandle:
        for listitem in imdb_id_to_delete:
            filehandle.write('%s\n' % listitem)
    # %%
    #escribir imdb_id_to_add
    with open('imdb_id_to_add.txt', 'w') as filehandle:
        for listitem in imdb_id_to_add:
            filehandle.write('%s\n' % listitem)

    # %%
    list_imdb = []
    column_name = ['imdbID', 'imdb_movie']
    for id in clean_imdb_id:
        list_imdb.append([id, ia.get_movie(id)])
        print(f'Se cargó correctamente el id: {id}')

    df_dict = pd.DataFrame(list_imdb, columns=column_name)
    # %%

    #escribir imdb_id_to_delete
    with open('list_imdb.txt', 'w') as filehandle:
        for listitem in list_imdb:
            filehandle.write('%s\n' % listitem)
    # %%
    # define an empty list
    list_imdb_temp = []

    with open('list_imdb.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            movie = line[:-1]

            # add item to the list
            list_imdb_temp.append(movie)
    # %%
#############################################
###
### TODO:
### 1.  Create feature to have unique codes
###     to the files (delete and add)
### 2.  Function to load the delete and add
###     and the search
### 3.  Manejo de errores para no hacerlo manual
### 4.  Manejo de errores no todos van a tener date on air
### 5.  en el dataframe el imdbID siempre es con 0
###
#############################################
    # %%

    # %%
list_dict_movie_detail = []
for imdb_movie in list_imdb:
    # distributors,genres,imdbID,kind,languages,original air date,plot,votes
    title = imdb_movie[1]['title']
    try:
        distributors = imdb_movie[1]['distributors'][0]['name']
    except:
        distributors = None
    genres = imdb_movie[1]['genres']
    kind = imdb_movie[1]['kind']
    languages = imdb_movie[1]['languages']
    year =     imdb_movie[1]['year']
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
        imdbID = imdb_movie[0],
        #imdb_movie = imdb_movie[1],
        title = title,
        distributors = distributors,
        year = year,
        plot = plot,
        votes = votes,
        original_title = original_title,
        writer = writer,
        runtimes = runtimes,
        countries = countries,
        original_air_date = original_air_date,
        rating = rating,
        director = director
         ))

    df = pd.DataFrame.from_dict(list_dict_movie_detail)
    #https://scrapsfromtheloft.com/stand-up-comedy-scripts/

    # %%
    df.to_json("dataframe_movie.json")
    df = pd.read_json(r'dataframe_movie.json')

    df['demographic_votes'] = None

    df['imdbID'].apply(lambda x: ia.get_movie_vote_details(x)['data']['demographics'])

    # %%
    lista = []
    for i in df['imdbID']:
        lista.append(ia.get_movie_vote_details(i)['data']['demographics'])
        print(f'df.loc[df[(df.imdbID=={i})][\'imdbID\'].index[0],\'imdbID\'] = \'0{i}')
    # %%
    #
    # #Exception
    # while True:
    #     try:
    #         paper_tiger = ia.get_movie('11318624')
    #     except:
    #         print('Hola mundu')
    #     print('ESO!')
