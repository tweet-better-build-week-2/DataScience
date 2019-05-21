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

def process_tweet(tweet):
  '''converts a single tweet into a form understandable by predictor.'''
  '''returns overall sentiment score between -1 and 1'''
  
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
