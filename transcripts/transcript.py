    # %%

from bs4 import BeautifulSoup
import requests
import pandas as pd


    # %%
def extract_transcript_urls():
    # %%
    url = 'https://scrapsfromtheloft.com/stand-up-comedy-scripts/'
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')
    # time.sleep(1)  # For some reason we have to wait
    cards_shows = soup.find_all('div', class_='elementor-post__text')
    stand_up_list = [stand_up.find('a').get('href').strip('/') for stand_up in cards_shows]
    # %%
    return stand_up_list

    # %%


def extract_transcript_urls(stand_up_list):
    # %%
    proto_df_show = []
    for stand_up in stand_up_list:
        html_text = requests.get(stand_up).text
        soup = BeautifulSoup(html_text, 'lxml')
        # time.sleep(1)  # For some reason we have to wait
        text_body = soup.find_all('div', class_='elementor-element elementor-element-74af9a5b elementor-widget '
                                                'elementor-widget-theme-post-content')
        # stand_up_list = [stand_up.find('a').get('href').strip('/') for stand_up in cards_shows]
        comedian = soup.find_all('a', class_='elementor-post-info__terms-list-item')[1].text
        title = soup.find('h1', class_='elementor-heading-title elementor-size-default').text
        strong_text = text_body[0].find_all('strong')
        raw_text = text_body[0].find_all('p')
        transcript = f''
        for paragraph in raw_text:
            if paragraph not in strong_text:
                transcript = transcript + paragraph.text + '\n'

        proto_df_show.append(dict(title=title, comedian=comedian, transcript=transcript))

    test = pd.DataFrame(proto_df_show)
    # %%
    return pd.DataFrame(proto_df_show)

