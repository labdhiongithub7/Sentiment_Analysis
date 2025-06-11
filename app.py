import os
import pickle
import joblib
import re
from flask import Flask, request, jsonify, render_template
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the model
pickled_model = pickle.load(open('regmodel.pkl', 'rb'))
bow_vectorizer = joblib.load('bow_vectorizer.joblib')

# Define the preprocessing function
stemmer = PorterStemmer()

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_review(review):
    review = remove_pattern(review, "@[\w]*")
    review = re.sub("[^a-zA-Z#]", " ", review)
    review = remove_emojis(review)
    review = review.encode('ascii', 'ignore').decode('ascii')
    review = " ".join([w for w in review.split() if len(w) > 3])
    review = " ".join([stemmer.stem(w) for w in review.split()])
    return review

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_review(text)
        review_bow = bow_vectorizer.transform([text])
        prediction = pickled_model.predict(review_bow)
        sentiment = 'positive' if prediction == 0 else 'negative'
        print(prediction)
        return render_template('home.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)


