# Import libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

#########################
def normalize_vector(v):
#######################
    if len(v.shape) == 2:
        return v/np.linalg.norm(v,axis=1).reshape(-1,1)
    else:
        return v/np.linalg.norm(v)
    
#########################
def plot_vectors(*vectors,plot_difference=False):
#################################
    # Stack vectors in matrix
    vecs = np.zeros((1,len(vectors[0])))
    
    for i, vec in enumerate(vectors):
        vecs = np.vstack((vecs,vec))
        
    # Compute difference between vectors
    if plot_difference:
        for i in range(len(vectors)-1):
            for j in range(i+1,len(vectors)):
                
                diff = vectors[j] - vectors[i]
                
                vecs = np.vstack((vecs,diff))
    
    # Remove dummy vector at first row
    vecs = vecs[1:,:]
    #print(vecs)
    
    # Compute x_lim and y_lin
    xy_limits = vecs.max(axis=0)
    x_lim = xy_limits[0]
    y_lim = xy_limits[1]
    #print(x_lim,y_lim)
        
    # Plot vectors
    ax = plt.figure(figsize=(8,5)).gca()
    
    arrow_dict  = {'head_width':0.008*x_lim, 'head_length':0.04*y_lim,'fc':'blue'}
    arrow_dict_d = {'fc':'grey', 'ec':'grey','linestyle':"dashed"}
    
    for i, vec in enumerate(vectors):
        # Plot arrow : (x-starting point, y-starting point, x-length,y_length, **kwargs)
        ax.arrow(0,0,vec[0],vec[1],**arrow_dict);
        
        #Annotate
        text = 'doc'+str(i+1)
        ax.annotate(text,(vec[0]+0.02*x_lim,vec[1]+0.02*y_lim))
    
    # Plot difference
    if plot_difference:
        for i in range(len(vectors)-1):
            for j in range(i+1,len(vectors)):
                
                diff = vectors[j] - vectors[i]
                ax.arrow(vectors[i][0],vectors[i][1],diff[0],diff[1],**arrow_dict_d);

    # Define limits for x/y-axes
    ax.set_xlim(-0.02*x_lim,x_lim+0.2*x_lim)
    ax.set_ylim(-0.02*y_lim,y_lim+0.2*y_lim)

    font_dict = {'fontsize':16,'fontname':'Arial','style':'italic','weight':'bold'}
    ax.set_xlabel('kindle',color='b',**font_dict)
    ax.set_ylabel('love',color='b',**font_dict);

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

#######################
# Plot cosine similarity using heatmaps

def plot_similarity(features):
#################################
    plt.figure(figsize=(20,20))
    corr = features 
    mask = np.triu(np.ones_like(corr, dtype=bool))

    g = sns.heatmap(
        corr,
        vmin=0,
        vmax=features.max().max(),
        cmap= "YlOrRd"
    )
    g.set_title("Semantic Textual Similarity")

##################
def visualize_bow_embeddings(X1,X2,df,label):
########################   
    title_specs = {'fontsize':16} #,'fontweight':'bold'}
    label_specs = {'fontsize':14}
    ticks_specs = {'fontsize':13}

    fig, axes = plt.subplots(1,2,figsize=(12,5))

    idx = df['semantic_category'] == label

    axes[0].scatter(X1[idx,0],X1[idx,1],color="none", edgecolor='m',label=label);
    axes[0].scatter(X1[~idx,0],X1[~idx,1],color="none", edgecolor='grey',alpha=0.3,label=None);
    axes[0].set_xlabel('TSNE 1',**label_specs)
    axes[0].set_ylabel('TSNE 2',**label_specs)
    axes[0].set_title('Bag-of-words',**title_specs)

    axes[1].scatter(X2[idx,0],X2[idx,1],color="none", edgecolor='b',label=label);
    axes[1].scatter(X2[~idx,0],X2[~idx,1],color="none", edgecolor='grey',alpha=0.3,label=None);
    axes[1].set_title('Embeddings',**title_specs)
    axes[1].set_xlabel('TSNE 1',**label_specs)
    axes[1].set_ylabel('TSNE 2',**label_specs)
    
    if label == 'movies':
        add_text(X1,idx=51,text='Star Trek', ax = axes[0]) #x=0.1)
        add_text(X1,idx=124,text='Night at the museum', ax = axes[0]) #,y=0.2)
        
        add_text(X2,idx=51,text='Star Trek', ax = axes[1], y=0.8)
        add_text(X2,idx=124,text='Night at the museum', ax = axes[1]) #,y=0.2)
        
    elif label == 'electronic devices':
        add_text(X1, 76,text='Canon EOS', ax = axes[0])
        add_text(X1, 267,text='Canon 40D', ax = axes[0])

        add_text(X2, 76,text='Canon EOS', ax = axes[1])
        add_text(X2, 267,text='Canon 40D', ax = axes[1])
        
    elif label == 'politics':
        add_text(X1, 375,text='Obama', ax = axes[0]) 
        add_text(X1, 86,text='North Korea', ax = axes[0])
        add_text(X1, 157,text='China', ax = axes[0]) #,y=0.1)
        add_text(X1, 247,text='Clinton', ax = axes[0]) #,x=0.2,y=-0.2)
        add_text(X1, 484,text='Iran', ax = axes[0]) #,x=-0.7)
        
        add_text(X2, 375,text='Obama', ax = axes[1]) 
        add_text(X2, 86,text='North Korea', ax = axes[1])
        add_text(X2, 157,text='China', ax = axes[1],y=-0.5)
        add_text(X2, 247,text='Clinton', ax = axes[1]) #,x=0.2,y=-0.2)
        add_text(X2, 484,text='Iran', ax = axes[1]) #,x=-0.7)
        
    elif label == 'Nike':
        add_text(X1, 73,text='Nike', ax = axes[0])
        add_text(X2, 73,text='Nike', ax = axes[1])
        
    elif label == 'mobile devices':
        add_text(X1, 227,text='iPhone', ax = axes[0])
        add_text(X1, 23,text='iPhone app', ax = axes[0]) 
        
        add_text(X2, 227,text='iPhone', ax = axes[1])
        add_text(X2, 23,text='iPhone app', ax = axes[1]) 
        
    elif label == 'Twitter':
        add_text(X1, 464,text='tweet', ax = axes[0])
        add_text(X1, 8,text='Twitter', ax = axes[0])
        add_text(X1, 60,text='Twitter API', ax = axes[0])

        add_text(X2, 464,text='tweet', ax = axes[1])
        add_text(X2, 8,text='Twitter', ax = axes[1])
        add_text(X2, 60,text='Twitter API', ax = axes[1])
        
    elif label == 'sports':
        add_text(X1, 19,text='Lebron', ax = axes[0])
        add_text(X1, 119,text='Lakers', ax = axes[0])
        add_text(X1, 171,text='NCAA', ax = axes[0])
        add_text(X1, 207,text='All-Star basket', ax = axes[0])
        add_text(X1, 404,text='NY Yankees', ax = axes[0])
        
        add_text(X2, 19,text='Lebron', ax = axes[1])
        add_text(X2, 119,text='Lakers', ax = axes[1])
        add_text(X2, 171,text='NCAA', ax = axes[1])
        add_text(X2, 207,text='All-Star basket', ax = axes[1], y = -1.4)
        add_text(X2, 404,text='NY Yankees', ax = axes[1])
        
    elif label == 'IT': 
        add_text(X1, 7,text='Jquery', ax = axes[0])
        add_text(X1, 480,text='LaTeX', ax = axes[0])
        add_text(X1, 319,text='λ-calculus', ax = axes[0])
        
        add_text(X2, 7,text='Jquery', ax = axes[1])
        add_text(X2, 480,text='LaTeX', ax = axes[1])
        add_text(X2, 319,text='λ-calculus', ax = axes[1])
        
    elif label == 'books':
        add_text(X1, 1,text='kindle2', ax = axes[0])
        add_text(X1, 0,text='kindle2', ax = axes[0])
        add_text(X1, 57,text='malcolm gladwell book', ax = axes[0])
        add_text(X1, 106,text='jQuery book', ax = axes[0])      
        
        add_text(X2, 1,text='kindle2', ax = axes[1])
        add_text(X2, 0,text='kindle2', ax = axes[1], y=-1.)
        add_text(X2, 57,text='malcolm gladwell book', ax = axes[1])
        add_text(X2, 106,text='jQuery book', ax = axes[1])


    plt.legend()
    plt.tight_layout();
    
    
##########
def add_text(embedding,idx,text,ax,x=0.0,y=0.0):
##################
    ax.annotate(text,(embedding[idx,0]+x,embedding[idx,1]+y),fontsize=12)
    