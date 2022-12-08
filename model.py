import re
import emoji
import demoji
import nltk
import pandas as pd
from emoji import emojize, UNICODE_EMOJI
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC

#nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')

df1 = pd.read_csv('labeled 5000.csv', low_memory=False)
slang_dict = pd.read_csv('colloquial-indonesian-lexicon.csv', usecols=['slang','formal'], low_memory=False)
slang_dict = dict(zip(slang_dict['slang'], slang_dict['formal']))
list_stopwords= set(list_stopwords)

class prep:
  def __init__(self, text):
    self.text= text
  def preprocessing(self):
  
    emoticon= (''.join(w for w in self.text if w in emoji.UNICODE_EMOJI['en']))
    retweet = re.findall('(rt @\w+:)', self.text)
    mention = re.findall('(@\w+)', self.text)
    hashtag = re.findall('(#\w+)', self.text)

    p1=self.text.lower()
  
    p2= re.sub('rt @\w+: ','', p1)      #rt
    p2= re.sub('@\w+','', p2)           #mention
    p2= re.sub('(#\w+)','', p2)         #hashtag

    p3= re.sub('http\S+', '', p2)       #URLs
    p3= re.sub(r'\n+', ' ', p3)         #Whitespaces
    p3= re.sub("'", '', p3)             #'
    p3= re.sub('[^\w\s]', ' ', p3)      #Punctuations
    p3= re.sub('\d', ' ', p3)           #Digits
    p3= re.sub('a{3,}', 'a',p3)         #aaaaaaa
    p3= re.sub('\s+', ' ', p3)          #Overspacings
    p3= re.sub('(^\s)|(\s$)', '', p3)   #Space on the start/end of string

    p4= re.split(r'\s', p3)
    p4= str(' '.join(slang_dict.get(word, word) for word in p4))

    st_factory= StemmerFactory()
    stemmer= st_factory.create_stemmer()
    p5= stemmer.stem(p4)

    sr_factory= StopWordRemoverFactory()
    remover= sr_factory.create_stop_word_remover()
    p6= remover.remove(p5)
    p6= re.split(r'\s', p6)
    token= [word for word in p6 if word not in list_stopwords]
    cleantext=' '.join(word for word in token)

    return cleantext, retweet, mention, hashtag, emoticon

  def nbg_classifier(self):
    tfidf_vect = TfidfVectorizer(max_features=mf, ngram_range=(1,2), norm=norm, max_df=mdf, use_idf=False)
    nb_gaussian_classifier = GaussianNB(var_smoothing=vs)

    x_train_tfidf = tfidf_vect.fit_transform(train_x)
    text_tfidf = tfidf_vect.transform(self)

    start_time = time.time()

    x_train_tfidf = x_train_tfidf.toarray()
    text_tfidf = text_tfidf.toarray()

    nbg = nb_gaussian_classifier.fit(x_train_tfidf, train_y)
  
    y_pred = nbg.predict(text_tfidf)
    ytrain_pred = nbg.predict(x_train_tfidf)
    y_pred_pr = nbg.predict_proba(text_tfidf)

    return y_pred, ytrain_pred, y_pred_pr