#!/usr/bin/env python

'''
GA Data Science Q2 2016

Code walk-through 6: Web scraping and APIs

* Web scraping using BeautifulSoup
* Accessing APIs
'''

import os
import re
import sys

from time import sleep

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup

'''
Web scraping using BeautifulSoup
'''

# Read HTML
html = requests.get('http://www.imdb.com/title/tt2084970/').text

# Parse HTML into a BeautifulSoup object
soup = BeautifulSoup(html, 'lxml')

# Retrieve the title
soup.find(name='h1')
soup.find(name='h1').find(text=True, recursive=False)
soup.find(name='h1').find(text=True, recursive=False).strip()

# Retrieve the genre(s)
soup.find_all(name='span', attrs={'itemprop': 'genre'})
[ x.text for x in soup.find_all('span', itemprop='genre') ]

# Retrieve the description
soup.find('div', itemprop='description').text.strip()

# Retrieve the duration (in minutes)
int(re.findall(r'(\d+)', soup.find('time', itemprop='duration')['datetime'])[0])

# Retrieve the content rating
soup.find('meta', itemprop='contentRating')['content']

# Retrieve the rating
float(soup.find('span', itemprop='ratingValue').text)

# Retrieve the rating and number of reviews
soup.find('div', class_='ratingValue').strong['title']
soup.find('div', 'ratingValue').strong['title']
rating, n = re.findall(r'^([\d\.]+).+?([\d,]+)', soup.find('div', 'ratingValue').strong['title'])[0]
rating = float(rating)
n = int(n.replace(',', ''))

# Define a function to do all of the above given an IMDb ID
def scrape_film_info(imdb_id):
    html = requests.get('http://www.imdb.com/title/' + imdb_id).text
    soup = BeautifulSoup(html, 'lxml')
    info = {}
    info['title'] =\
        soup.find('h1').find(text=True, recursive=False).strip()
    info['genres'] =\
        [ x.text for x in soup.find_all('span', itemprop='genre') ]
    info['description'] =\
        soup.find('div', itemprop='description').text.strip()
    info['duration'] =\
        int(re.findall(r'(\d+)', soup.find('time', itemprop='duration')['datetime'])[0])
    info['content_rating'] =\
        soup.find('meta', itemprop='contentRating')['content']
    rating, n =\
        re.findall(r'^([\d\.]+).+?([\d,]+)',\
                   soup.find('div', 'ratingValue').strong['title'])[0]
    info['rating'] = float(rating)
    info['n'] = int(n.replace(',', ''))
    return info

# Test it!
scrape_film_info('tt2084970')

# Get the 'Top 250 as rated by IMDb Users' list
soup = BeautifulSoup(requests.get('http://www.imdb.com/chart/top').text, 'lxml')

# Retrieve the list of IMDb IDs
tmp = soup.find_all(name='td', attrs={'class': 'titleColumn'})
imdb_ids = [ re.findall(r'/(tt[0-9]+)/', x.a['href'])[0] for x in tmp ]

# Keep only the top 10 films
imdb_ids = imdb_ids[:10]

# Retrieve the information for each film
films = []
for imdb_id in imdb_ids:
    films.append(scrape_film_info(imdb_id))
    sleep(1)

# Make sure that we successfully retrieved information for all the films
assert(len(imdb_ids) == len(films))

# Convert to a DataFrame
films = pd.DataFrame(films, index=imdb_ids)

'''
Accessing APIs
'''

# Send API request
req = requests.get('http://www.omdbapi.com/?i=tt2084970&type=movie&r=json')

# Check HTTP status code (2xx = success, 4xx = client error, 5xx = server error)
req.status_code

# Get the raw response
req.text

# Decode the JSON response into a dictionary
req.json()

# Define a function to retrieve the release date
def scrape_release_date(imdb_id):
    req = requests.get('http://www.omdbapi.com/?i=' + imdb_id + '&type=movie&r=json')
    info = req.json()
    if info['Response'] == 'True':
        return info['Released']
    else:
        return None

# Test it!
scrape_release_date('tt2084970')

# Create a new column 'release_date' in 'films' and set it to missing
films['release_date'] = np.nan

# Update the release date
for idx, row in films.iterrows():
    films.loc[idx,'release_date'] = scrape_release_date(row.name)
    sleep(1)

# Convert 'release_date' to `datetime`
films.release_date = pd.to_datetime(films.release_date)

