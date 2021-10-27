import re
import string
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

emoticons = {}
emoticons['positive'] = ['x-d', ':-)', ":'-)", '8)', ':o)', '>:)', ':-p', ';)', '8-d', ':p', 'xp', ':^)',
                         ':-b', '8d', ':}', '=)', '=-d', ':)', '=p', ':3', ':b', ':^*', '=d', ':]', ':>', '=-3', 'x-p', ':d', '>:-)',
                         ":')", ':*', '<3', '=]', ':-d', 'xd', '=3', ':c)', ':-))', '>;)', '>:p']
emoticons['negative'] = [':\\', ':l', ':<', ':s', ':-||', ';(', ":'-(", '>.<', '=\\', ':{', '=/', ':(', ':c', '>:[', ':-c', '>:(', '>:/',
                         ':-(', '=l', '>:\\', ':-/', ":'(", ':[', ':@', ':-<', ':-[']


def process_tweet(tweet, remove_emoticons=False):
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

    # Removing positive and negative emoticons
    if remove_emoticons:
        tweet_tokens_cleaned = [token for token in tweet_tokens_cleaned
                                if token not in (emoticons['positive'] + emoticons['negative'])]

    # Stemming
    stemmer = PorterStemmer()

    tweet_stems = [stemmer.stem(token) for token in tweet_tokens_cleaned]

    return tweet_stems


def build_freqs(tweets, labels, remove_emoticons = False):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        labels: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(labels, tweets):
        for word in process_tweet(tweet, remove_emoticons):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def build_freqs_alt(tweets, labels, remove_emoticons = False):
    '''An Alternative implementation using the Counter object from the 
    collections module. Here we only loop once over all the tweets then update
    the Counter with new counts. Time execution is slightly longer than the
    function build_freqs.
    '''

    freqs = Counter()
    for tweet, label in zip(tweets, labels):
        token_label = [(token, label) for token in process_tweet(tweet, remove_emoticons)]
        freqs.update(Counter(token_label))
    return freqs
