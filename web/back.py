from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

from model import prep
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import make_pipeline

nbg = pickle.load(open('model nbg 12.pkl','rb'))
nbm = pickle.load(open('model nbm 12.pkl','rb'))
tfidf_nbg = pickle.load(open('model tfidf nbg.pkl','rb'))
tfidf_nbm = pickle.load(open('model tfidf nbm.pkl','rb'))

app =Flask(__name__)

@app.route('/', methods=['POST','GET'])
def main():
    if request.method == 'POST':
        inp = str(request.form.get('input_sentiment'))
        #classifier = SentimentIntensityAnalyzer()
        #score = classifier.polarity_scores(inp)
        #if score['neg'] != 0:
        #    return render_template('home.html', message='negative')
        #else:
        #    return render_template('home.html', message='positive')
        text = prep(inp)
        cleantext, retweet, mention, hashtag, emoticon = text.preprocessing()
        text_tfidf = tfidf_nbm.transform([cleantext])
        text_tfidf = text_tfidf.toarray()
        y_pred = nbm.predict(text_tfidf)
        if cleantext != '':
            if y_pred == 1:
                return render_template('home.html', message='This is a nice positive sentiment <3', cleantext='CleanText: [%s]' %cleantext, text='"%s"' %inp, mention='Retweet & Mentions: %s' %mention, hashtag='Hashtag: %s' %hashtag, emoticon='Emojis: %s' %emoticon)
            else:
                return render_template('home.html', message='This is an awfully negative sentiment </3', cleantext='CleanText: [%s]' %cleantext, text='"%s"' %inp, mention='Retweet & Mentions: %s' %mention, hashtag='Hashtag: %s' %hashtag, emoticon='Emojis: %s' %emoticon)
        else:
           return render_template('home.html', message='Can not classify this sentiment, perhaps it is a neutral sentiment', cleantext='CleanText: [%s]' %cleantext, text='"%s"' %inp, mention='Retweet & Mentions: %s' %mention, hashtag='Hashtag: %s' %hashtag, emoticon='Emojis: %s' %emoticon)     
    return render_template('home.html')

if __name__=='__main__':
    app.run()
