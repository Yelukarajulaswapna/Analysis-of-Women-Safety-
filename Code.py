from textblob import TextBlob
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import re
import string
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter, defaultdict
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import re
import string
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter, defaultdict
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
tweets_list = []
clean_list = []
train = pd.read_csv("dataset/MeToo_tweets.csv", encoding='iso-8859-1')
train
train.info()
train.isnull().sum()
train.describe()
train
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
# '[%s]' % re.escape(string.punctuation),' ' - replace punctuation with white space
# .lower() - convert all strings to lowercase 
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
 # Remove all '\n' in the string and replace it with a space
remove_n = lambda x: re.sub("\n", " ", x)
# Remove all non-ascii characters 
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
# Apply all the lambda functions wrote previously through .map on the comments column
train['Text'] = 
train['Text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_as
cii)
train
def polarity(text):
 testimonial = TextBlob(text)
 polarity = testimonial.sentiment.polarity
 return polarity
def subjectivity(text):
 testimonial = TextBlob(text)
 subjectivity = testimonial.subjectivity
 return subjectivity
def senti(text, polarity_threshold=0.2):
 testimonial = TextBlob(text)
 senti = testimonial.sentiment.polarity
 if senti >= polarity_threshold:
 return 'Positive'
 elif np.abs(senti) < polarity_threshold:
return 'Neutral'
 else:
 return 'Negative'
train['polarity'] = train['Text'].apply(lambda x: polarity(x))
train['subjectivity'] = train['Text'].apply(lambda x: subjectivity(x))
train['sentiment'] = train['Text'].apply(lambda x: senti(x))
train
train['score'] = train['sentiment'].map({'Neutral': 0, 'Positive' : 1, 'Negative' : 2})
train
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(train['Text']).toarray()
 y = train['score']
from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
 from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
 from sklearn import metrics
 def evaluate(model, X_train, X_test, y_train, y_test):
 y_test_pred = model.predict(X_test)
 y_train_pred = model.predict(X_train)
 print("TRAINING RESULTS: \n===============================")
 clf_report=pd.DataFrame(classification_report(y_train,y_train_pred,output_dict=True))
 print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
 print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
 print(f"CLASSIFICATION REPORT:\n{clf_report}")
 print("TESTING RESULTS: \n===============================")
 clf_report=pd.DataFrame(classification_report(y_test,y_test_pred, output_dict=True))
 print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
 print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
 print(f"CLASSIFICATION REPORT:\n{clf_report}")
 from sklearn import svm
sv = svm.SVC()
sv.fit(x_test, y_test)
svm_score = sv.score(x_test, y_test) * 100
evaluate(sv,x_train, x_test, y_train, y_test)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
dt_score = DT.score(x_test, y_test) * 100
evaluate(DT,x_train, x_test, y_train, y_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
clf_score = clf.score(x_test, y_test) * 100
evaluate(clf,x_train, x_test, y_train, y_test)
from flask import Flask,request, url_for, redirect, render_template
import pandas as pd
import pickle
import numpy as np
import sqlite3
from textblob import TextBlob
app = Flask(__name__)
tweets_list = []
clean_list = []
@app.route('/')
def hello_world():
 return render_template("home.html")
@app.route('/logon')
def logon():
return render_template('signup.html')
@app.route('/login')
def login():
return render_template('signin.html')
@app.route('/note')
def note():
return render_template('notebook.html')
def tweetCleaning(doc):
 tokens = doc.split()
 table = str.maketrans('', '', punctuation)
 tokens = [w.translate(table) for w in tokens]
 tokens = [word for word in tokens if word.isalpha()]
 stop_words = set(stopwords.words('english'))
 tokens = [w for w in tokens if not w in stop_words]
 tokens = [word for word in tokens if len(word) > 1]
 tokens = ' '.join(tokens) #here upto for word based
 return tokens
def clean():
 text.delete('1.0', END)
 clean_list.clear()
 for i in range(len(tweets_list)):
 tweet = tweets_list[i]
 tweet = tweet.strip("\n")
 tweet = tweet.strip()
 tweet = tweetCleaning(tweet.lower())
 clean_list.append(tweet)
@app.route("/signup")
def signup():
 username = request.args.get('user','')
 name = request.args.get('name','')
 number = request.args.get('mobile','')
 email = request.args.get('email','')
 password = request.args.get('password','')
 con = sqlite3.connect('signup.db')
 cur = con.cursor()
 cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES 
(?, ?, ?, ?, ?)",(username,email,password,number,name))
con.commit()
 con.close()
 return render_template("signin.html")
 @app.route("/signin")
 def signin():
 mail1 = request.args.get('user','')
 password1 = request.args.get('password','')
 con = sqlite3.connect('signup.db')
 cur = con.cursor()
 cur.execute("select `user`, `password` from info where `user` = ? AND `password` = 
?",(mail1,password1,))
 data = cur.fetchone()
 if data == None:
 return render_template("signin.html") 
elif mail1 == 'admin' and password1 == 'admin':
 return render_template("index.html")
 elif mail1 == str(data[0]) and password1 == str(data[1]):
 return render_template("index.html")
 else:
 return render_template("signup.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
 if request.method == 'POST':
 text = request.form['message']
 blob = TextBlob(text)
 if blob.polarity <= 0.2:
 prediction = 0
 elif blob.polarity > 0.2 and blob.polarity <= 0.5:
 prediction = 2
 elif blob.polarity > 0.5:
 prediction = 1
 if prediction == 0:
return render_template('result.html',pred=f'Negative')
 elif prediction == 1:
 return render_template('result.html',pred=f'Positive')
 elif prediction == 2:
 return render_template('result.html',pred=f'NEUTRAL')
@app.route('/index')
def index():
return render_template('index.html')
if __name__ == '__main__':
 app.run(debug=True)
