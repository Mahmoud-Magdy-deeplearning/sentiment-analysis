import re                                  # library for regular expression operations
import string                              # for string operations
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
nltk.download('stopwords')

def process_tweet(tweets):

    for tweet in tweets:
        # remove old style retweet text "RT"
        tweet2 = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)

        # remove hashtags
        # only removing the hash # sign from the word
        tweet2 = re.sub(r'#', '', tweet2)

        # instantiate tokenizer class
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                       reduce_len=True)

        # tokenize tweets
        tweet_tokens = tokenizer.tokenize(tweet2)
        #Import the english stop words list from NLTK
        stopwords_english = stopwords.words('english') 
        
        tweets_clean = []

        for word in tweet_tokens: # Go through every word in your tokens list
            if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
                tweets_clean.append(word)

        tweets_stem = [] 

        for word in tweets_clean:
            stem_word = stemmer.stem(word)  # stemming word
            tweets_stem.append(stem_word)  # append to the list
        
        return tweets_clean
    
           
           
def build_freqs(tweet):
  
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

