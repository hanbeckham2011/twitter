import json
import tweepy
import pandas as pd

def clean_tweet(tweet):
    tweet = tweet.text.encode("ascii", "ignore").decode()
    tweet = tweet.replace('\n', '').replace('RT', '').replace('#', '').lstrip('RT')
    while '@' in tweet: # remove usernames
        previous = tweet
        tweet = tweet[tweet.index(' '):]
        if tweet == previous:
            break
    return tweet

with open('credentials.txt') as f:
    data = f.read()
credentials = json.loads(data)

search_query = 'bad happy'

client = tweepy.Client(bearer_token=credentials['bearer_token'],
                       consumer_key=credentials['consumer_key'],
                       consumer_secret=credentials['consumer_secret'],
                       access_token=credentials['access_token'],
                       access_token_secret=credentials['access_secret'])

tweets = client.search_recent_tweets(query=search_query, max_results=10)[0]

data = []
for tweet in tweets:
    text = clean_tweet(tweet)
    data.append(text)

df = pd.DataFrame(data, dtype='string')
df.to_csv('./pulled_tweets.csv', sep=',', header=False)