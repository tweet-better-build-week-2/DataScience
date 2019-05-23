'''Functions for actions in the Tweet Better website.'''

#imports
import string
from string import punctuation
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer


def process_tweet(tweet):
  '''converts a single tweet into a form understandable by predictor.'''
  '''returns sentiment score between 0 and 100'''
  
  #nltk stopwords are words too common to be useful
  #not stopwords as in profanity
  stop_words = stopwords.words('english')
  
  #convert to string
  tweet = str(tweet)
  
  #remove urls
  url = re.compile('http.+? ')
  tweet = re.sub(url, '', tweet)
  
  #remove tagged users
  user = re.compile('@.+? ')
  tweet = re.sub(user, "", tweet)
  
  #clean html tags and literals
  tag = re.compile('<.*?>')
  tweet = re.sub(tag, '', tweet)
  tweet = tweet.replace(r"\n", " ")
  tweet = tweet.replace(r"b'", "")
  tweet = tweet.replace(r"\r", "")
  
  #temp. tokenize to clean on word level
  words = word_tokenize(tweet)
  
  #remove spaces (will be added in again)
  words = [word.strip() for word in words]
  
  #lowercase
  words = [word.lower() for word in words]
  
  #remove stopwords
  words = [word for word in words if not word in stop_words]
  
  #lemmatize (remove prefixes and suffixes)
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words]
  fin_tweet = " ".join(words)
  
  #predict
  sid = SentimentIntensityAnalyzer()
  scores = sid.polarity_scores(fin_tweet)
  score = scores['compound']
  
  return (score/2 + 1/2) * 100


def score_input(tweet):
  '''for use when a user types a new tweet.'''
  '''returns a statement depending on score.'''
  score = process_tweet(tweet)
  if score < float(25):
    return print("Your tweet might be more negative than it reads to you now.\nIt has a sentiment score of "+str(score)+"%.\nDo you still want to send the tweet or should it be drafted?")
  elif score > float(75):
    return print("You're improving Twitter one tweet at a time!\nThis tweet has a highly positive sentiment score of "+str(score)+"%.\nSend now?")
  else:
    return print("This is a pretty neutral tweet, with a sentiment score of "+str(score)+"%.\nSend now?")
  
  
def score_timeline(timeline):
  '''for use when a user requests to evaluate entire timeline.'''
  '''returns a list of scores (for graphing by time)'''
 scores = []
 for tweet in timeline:
   score = process_tweet(tweet)
   scores.append(score)
 return scores

def clean_tweet_regression(tweet):
  '''converts a single tweet into a form understandable by predictor.'''
  '''variation of existing cleaner to clean punct and'''
  '''return string rather than score'''
  
  #table that replaces punct with empty space
  table = str.maketrans('', '', punctuation)
  
  #nltk stopwords are words too common to be useful
  #not stopwords as in profanity
  stop_words = stopwords.words('english')
  
  #convert to string
  tweet = str(tweet)
  
  #remove urls
  url = re.compile('http.+? ')
  tweet = re.sub(url, '', tweet)
  
  #remove tagged users
  user = re.compile('@.+? ')
  tweet = re.sub(user, "", tweet)
  
  #clean html tags and literals
  tag = re.compile('<.*?>')
  tweet = re.sub(tag, '', tweet)
  tweet = tweet.replace(r"\n", " ")
  tweet = tweet.replace(r"b'", "")
  tweet = tweet.replace(r"\r", "")
  
  #temp. tokenize to clean on word level
  words = word_tokenize(tweet)
  
  #remove spaces (will be added in again)
  words = [word.strip() for word in words]
  
  #lowercase
  words = [word.lower() for word in words]
  
  #remove non alphanumeric characters
  words = [word.translate(table) for word in words]
  words = [word for word in words if word.isalpha()]
  
  #remove stopwords
  words = [word for word in words if not word in stop_words]
  
  #lemmatize (remove prefixes and suffixes)
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word) for word in words]
  fin_tweet = " ".join(words)
  
  return fin_tweet

def clean_timeline_regression(timeline):
  '''loop of clean_tweet_regression'''
  '''runs as part of the function regression'''
  clean = []
  for tweet in timeline:
    fin_tweet = clean_tweet_regression(tweet)
    clean.append(fin_tweet)
  return clean

def regression(tweets, likes, new_tweet):
  '''trains a ridge regression live with the users timeline.
     can accept any metadata variable that is a number.
     stretch goal: make a multivariate regression'''
  
  #get training data
  clean_tweets = clean_timeline_regression(tweets)
  vectorizer = CountVectorizer()
  vectorizer.fit(clean_tweets)
  word_counts = vectorizer.transform(clean_tweets)
  X = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names())
  X['target'] = likes
  y = X['target']
  X = X.drop(columns='target')
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)
  
  #fit Ridge; evaluate what alpha results in lowest loss
  alphas = []
  mses = []
  
  ridge_reg = Ridge().fit(X, y)
  
  for alpha in range(0, 200, 1):
      ridge_reg_split = Ridge(alpha=alpha).fit(Xtrain, ytrain)
      mse = mean_squared_error(ytest, ridge_reg_split.predict(Xtest))
      alphas.append(alpha)
      mses.append(mse)
  
  #lowest mse in list and its index
  min_index = mses.index(min(mses))
  
  #fit training data to min_index alpha
  ridge_reg_split = Ridge(alpha=min_index).fit(Xtrain, ytrain)
  
  #clean the new tweet
  cl = [clean_tweet_regression(new_tweet)]
  new_array = vectorizer.transform(cl).toarray()
  
  #return a prediction for new tweet
  pred = int(ridge_reg_split.predict(new_array))
  return pred
  
