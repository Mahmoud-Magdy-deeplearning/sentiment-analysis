from os import getcwd
import pandas as pd                          # Library for Dataframes 
import re                                    # library for regular expression operations
import string                                # for string operations
import nltk                                  # NLP toolbox
from nltk.corpus import twitter_samples 
import numpy as np                           # Library for math functions
from utils import process_tweet              # Import the process_tweet function
from utils import process_tweet, build_freqs # Our functions for NLP

class preprocessing:

    # select the lists of positive andpro negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # concatenate the lists, 1st part is the positive tweets followed by the negative
    tweets = all_positive_tweets + all_negative_tweets

    # make a numpy array representing labels of the tweets
    labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

    processed tweets = []
    for tweet in tweets: 
        processed tweets = process_tweet(tweet) # Preprocess a given tweet

    # create frequency dictionary
    freqs = build_freqs(processed tweets, labels)

    # list representing our table of word counts.
    # each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]pro
    data = []

    # loop through our selected words
    for word in keys:

        # initialize positive and negative counts
        pos = 0
        neg = 0

        # retrieve number of positive counts
        if (word, 1) in freqs:
            pos = freqs[(word, 1)]

        # retrieve number of negative counts
        if (word, 0) in freqs:
            neg = freqs[(word, 0)]

        # append the word counts to the table
        data.append([word, pos, neg])

    def get_preprocessed_data():
        return processed tweets
    
    def get_freqs_dict():
        return freqs 
        
    def get_freqs_table():
        return data
