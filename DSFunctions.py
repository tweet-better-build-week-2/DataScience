'''Functions for actions in the Tweet Better website.'''
'''TODO: make process_tweetob that takes tweets from API. return score[compound].'''
'''TODO: unicode cleaning code if necessary'''

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

def process_tweet(tweet):
  '''converts a single tweet into a form understandable by predictor.'''
  '''returns positive, neutral, negative score and message.'''
  
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
  
  #return a message based on compound sentiment
  if scores['compound'] < float(-0.50):
    return print("Your tweet might be more negative than it reads to you now.\nIt has an overall sentiment score of "+str(scores['compound'])+".\n"+str(scores)+"\nDo you still want to send the tweet or should it be drafted?")
  elif scores['compound'] > float(0.50):
    return print("You're improving Twitter one tweet at a time!\nThis tweet has a highly positive sentiment score of "+str(scores['compound'])+".\n"+str(scores)+"\nSend now?")
  else:
    return print("This is a pretty neutral tweet, with an overall sentiment of "+str(scores['compound'])+".\n"+str(scores)+"\nSend now?")
