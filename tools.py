# Import libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re, random, string
from collections import defaultdict
import nltk
import contractions
from nltk.tokenize import TweetTokenizer

# Gensim
import gensim
from gensim.models import Phrases

pd.set_option('display.max_colwidth',None)

######################################
def add_to_semantic_category(df):
######################################    
    '''
    Create "semantic" labels for tweets, based on selected keywords. 
    This will be useful for finding insights in plots in later notebooks. 

    For example, 
    - any tweet mentioning movie-related terms (e.g. "watch_night", "star_treck", "movi", etc.) -> category: "movies"
    - tweets mentioning politics-related terms (e.g. "north_korea", "obama", "pelosi", "bush", etc.) ->  category : "politics"
    - tweets mentionning sports-related terms (e.g. "lebron", "laker", "basebal", "fifa", etc.)  ->  category : "sports"
    
    And so on.
    '''

    # Define keywords to look for
    selected_keywords = ['gm',                                                  # general motors (car industry)
                         'nikon','canon',                                       # DSLR cameras
                         'time_warner','time-warn','comcast','espn',            # cable-tv
                         'night_museum','star_trek','see_star','movi',          # movies
                         'watch_night','see_night',                             # movies
                         'dentist','tooth',                                     # dentist
                         'lebron','laker','basebal','basketbal','fifa','ncaa',  # sports
                         'sport','yanke','roger',                               # sports
                         'nike',                                                # Nike
                         'twitter_api','twitter','tweet',                       # Twitter
                         'phone','android','iphon','tether',                    # mobile devices
                         'latex','jqueri','linux','wolfram',                    # IT (Information technology)
                         'lambda', 'classif',                                   # IT (Information technology)
                         'north_korea','obama','pelosi','bush','china',         # politics
                         'india','iran','irancrazi','us',                       # politics
                         'eat','ate','food','mcdonald','safeway',               # food
                         'san_francisco','montreal','east_palo',                # cities
                         'book','malcolm_gladwel','kindl',                       # books
                         'blog', 'new_blog',                                    # blogging
                         'bobbi_flay',                                          # Chef
                         'buffet','warren','warren_buffet'                      # Warren Buffett (American investor)
                        ]

    # Define semantic clusters
    categories_dict = {'gm':'car industry',                                     # general motors (car industry)

                       'nikon':'DSLR cameras',                                  # electronic devices
                       'canon':'DSLR cameras',                            # electronic devices

                       'time_warner':'cable TV','time-warn':'cable TV',         # cable-tv
                       'comcast':'cable TV','espn':'cable TV',                  # cable-tv

                       'night_museum':'movies','star_trek':'movies',            # movies
                       'see_star':'movies','movi':'movies',                     # movies
                       'watch_night':'movies','see_night':'movies',             # movies

                       'dentist':'dentist','tooth':'dentist',                   # dentist

                       'lebron':'sports','laker':'sports',                      # sports
                       'basebal':'sports','basketbal':'sports',                 # sports
                       'fifa':'sports','ncaa':'sports',                         # sports
                       'sport':'sports','yanke':'sports','roger':'sports',      # sports

                       'nike': 'Nike',                                          # Nike

                       'twitter_api':'Twitter','twitter':'Twitter',             # Twitter
                       'tweet':'Twitter',                                       # Twitter

                       'phone':'mobile devices','android':'mobile devices',     # mobile devices
                       'iphon':'mobile devices','tether':'mobile devices',      # mobile devices

                       'latex':'IT','jqueri':'IT','classif':'IT',               # IT (Information technology)
                       'linux':'IT','wolfram':'IT',                             # IT (Information technology)
                       'lambda':'IT',                                           # IT (Information technology)

                       'north_korea':'politics','obama':'politics',             # politics
                       'pelosi':'politics','bush':'politics',                   # politics
                       'china':'politics', 'india':'politics',                  # politics
                       'iran':'politics','irancrazi':'politics',                # politics
                       'us':'politics',                                         # politics

                       'eat':'food','ate':'food','food':'food',                 # food
                       'mcdonald':'food','safeway':'food',                      # food

                       'san_francisco':'cities','montreal':'cities',            # cities
                       'east_palo':'cities',                                    # cities

                       'kindl':'books',                                         # books
                       'book':'books','malcolm_gladwel':'books',                # books
                       'blog':'blogging', 'new_blog':'blogging',                # blogging
                       'bobbi_flay':"Bobby Flay",                               # Chef

                       'buffet':'Warren Buffett','warren':'Warren Buffett',     # Warren Buffett (American investor)
                       'warren_buffet':'Warren Buffett'                         # Warren Buffett (American investor)

                       }

    # Look for these keywords in tweets
    keywords = []

    for tweet in df['processed_tweet']:
        cat = 'unlabeled'
        for keyword in selected_keywords:
            if keyword in tweet.split():
                cat = keyword
        keywords.append(cat)

    # Add keywords to DataFrame
    df['keyword'] = keywords
    
    # Add tweet to semantic category based on found keyword
    df['semantic_category'] = df['keyword'].replace(categories_dict)
    
    # Remove column 'keyword'
    df.drop('keyword',axis=1,inplace=True)
    
    return df

##########################
def plot_most_frequent_terms(frequency_dict, terms_to_plot,add_to_title=None):
###############    
    '''
    Plots the most frequent collocations in corpus
    
    INPUTS:
    - frequency_dict:  dictionary, a dictionary mapping terms to raw counts
    - terms_to_plot :  integer, number of (most frequent) collocations to plot
    - add_to_title  :  string, complementary string for the plot title  
                       The title defaults to: "Top X terms" -> use add_to_title to complete
                       the title string (optional)
    
    OUTPUT:
    - Barplot of most frequent terms and their respective frequency
    
    '''
    
    # Barplot and font specifications
    barplot_specs = {'color':'mediumpurple','alpha':0.8,'edgecolor':'grey'}
    title_specs = {'fontsize':16} #,'fontweight':'bold'}
    label_specs = {'fontsize':14}
    ticks_specs = {'fontsize':13}
    
    title = 'Top '+str(terms_to_plot)+' terms'
    
    if add_to_title == None:
        title = title
    else:
        title = title + ' '+ str(add_to_title)
        
    ylabel = 'Counts'

    # Plot top terms and their frequency
    plt.figure(figsize=(18,3))
    sns.barplot(x = list(frequency_dict.keys())[0:terms_to_plot], y = list(frequency_dict.values())[0:terms_to_plot],**barplot_specs)
    plt.ylabel(ylabel,**label_specs)
    plt.title(title,**title_specs)
    plt.xticks(rotation=80,**ticks_specs)
    plt.yticks(**ticks_specs); 
    
    
    
# Create clean_tweet function
####################################
def clean_tweet_plot(tweet):
####################################    
#    import nltk, contractions

    # Import tokenizer
#    from nltk.tokenize import TweetTokenizer

    # Create an instance  of the tokenizer
    tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
    
    '''
    INPUT: 
    - tweet: raw text
    
    OUTPUT:
    - clean_tweet: cleaned text
    '''

    # Remove RT
    clean_tweet = re.sub(r'RT','',tweet)

    # Remove URL
    clean_tweet = re.sub(r'https?:\/\/[^\s]+','',clean_tweet)

    # Remove hash #
    clean_tweet = re.sub(r'#','',clean_tweet)
        
    # Remove twitter username
    clean_tweet = re.sub(r'@[A-Za-z]+','',clean_tweet)
    
    # Remove punctuation repetions (that are not removed by TweetTokenizer)
    clean_tweet = re.sub(r'([._]){2,}','',clean_tweet)
    
    # Case conversion
    clean_tweet = clean_tweet.lower()
    
    # Remove non-ascii chars
    clean_tweet = ''.join([c for c in str(clean_tweet) if ord(c) < 128])

    # Expand contractions
    clean_tweet = contractions.fix(clean_tweet)
    
    # Tokenize tweet
    tokens = tokenizer.tokenize(clean_tweet)

    # Join tokens in a single string to recreate the tweet
    clean_tweet = ' '.join([tok for tok in tokens])
    
    clean_tweet = re.sub(r'\s\.','.',clean_tweet)
    clean_tweet = re.sub(r'\s,',',',clean_tweet)
    clean_tweet = re.sub(r'\s!','!',clean_tweet)
    clean_tweet = re.sub(r'\s\?','?',clean_tweet)
    clean_tweet = re.sub(r'\$','',clean_tweet)
    clean_tweet = re.sub(r'\s+',' ',clean_tweet)
    
    clean_tweet = clean_tweet.strip()
    clean_tweet = re.sub(r'^[:-]','',clean_tweet)
    clean_tweet = clean_tweet.strip()
    
    short_text = clean_tweet.split()
    
    return ' '.join(short_text[:5])+'...'