# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/api',methods=['GET', 'POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Load the model
    model = SentimentIntensityAnalyzer()
    # Make prediction using model loaded from disk as per the data.
    prediction = model.polarity_scores(data['texts'])
    # prediction = data['texts']
    output = prediction['compound']
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)