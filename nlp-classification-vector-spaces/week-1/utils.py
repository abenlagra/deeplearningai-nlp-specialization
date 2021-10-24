import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    ''' This function processes the tweets in order to normalize them. In
    particular, the processing consists of:
    - Removing hyperlinks, twitter marks and styles
    - Tokenizing the Tweet
    - Removing stop words and punctuations
    - Stemming

    The function returns a list of stems for the cleaned tweet
    '''

    # Removing hyperlinks
    clean_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # Removing twitter marks
    clean_tweet = re.sub(r'#', '', clean_tweet)

    # Removing styke
    clean_tweet = re.sub(r'^RT[\s]+', '', clean_tweet)

    # Tokenizing the Tweet
    tokenizer = TweetTokenizer(preserve_case=False,
    strip_handles=True,
    reduce_len=True)

    tweet_tokens = tokenizer.tokenize(clean_tweet)

    # Removing stopwords and punctuations
    stopwords_english = stopwords.words('english')

    tweet_tokens_cleaned = [token for token in tweet_tokens
    if token not in stopwords_english
    and token not in string.punctuation]

    # Stemming
    stemmer = PorterStemmer()

    tweet_stems = [stemmer.stem(token) for token in tweet_tokens_cleaned]

    return tweet_stems
