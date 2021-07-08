#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


dataset = "C:\\Users\\Lampe\\Desktop\\sentiment analysis\\dataset\\dataset.csv"


# In[3]:


import chardet
with open(dataset, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# In[4]:


data=pd.read_csv(dataset,encoding='Windows-1252')


# In[5]:


data.head()


# In[6]:


#Basic visualization of the data
sns.countplot(data['Sentiment'])


# In[7]:


#Removing Punctuation
punc = string.punctuation
plist = punc
def rempunc(text):
    trans = str.maketrans('', '', plist)
    return text.translate(trans)
data['SentimentText']= data['SentimentText'].apply(lambda x: rempunc(x))


# In[8]:


data.head()
data


# In[9]:


#Removing Hashtags and Mentions
def removemenandhash(text):
    items = ['@','#']
    for separator in  string.punctuation:
        if separator not in items :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in items:
                words.append(word)
    return ' '.join(words)
data['SentimentText']= data['SentimentText'].apply(lambda x: removemenandhash(x))


# In[10]:


data


# In[11]:


print(data.loc[[9]])


# In[12]:


data


# In[13]:


#Removing stopwords and doing the other necessary preprocessing
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
  
VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

def prep(text):
  text = text.lower()
  temp_sent =[]
  words = nltk.word_tokenize(text)
  tags = nltk.pos_tag(words)
  for i, word in enumerate(words):
      if tags[i][1] in VERB_CODES: 
          lemmatized = lemmatizer.lemmatize(word, 'v')
      else:
          lemmatized = lemmatizer.lemmatize(word)
      if lemmatized not in stop_words and lemmatized.isalpha():
          temp_sent.append(lemmatized)
          
  finalsent = ' '.join(temp_sent)
  finalsent = finalsent.replace("n't", " not")
  finalsent = finalsent.replace("'m", " am")
  finalsent = finalsent.replace("'s", " is")
  finalsent = finalsent.replace("'re", " are")
  finalsent = finalsent.replace("'ll", " will")
  finalsent = finalsent.replace("'ve", " have")
  finalsent = finalsent.replace("'d", " would")
  return finalsent

data['SentimentText']= data['SentimentText'].apply(lambda x: prep(x))


# In[14]:


#Removing links and numbers
data['SentimentText'] = data['SentimentText'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')
data['SentimentText'] = data['SentimentText'].str.replace('\d+', '')
data['SentimentText'] = data['SentimentText'].str.replace(r'http?://[^\s<>"]+|www\.[^\s<>"]+', '')


# In[15]:


data


# In[16]:


tf = TfidfVectorizer(analyzer='word',min_df=0, stop_words='english')


# In[17]:


X = tf.fit_transform(data['SentimentText'])
Y = data.iloc[:,[1]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=50)


# In[18]:


X


# In[19]:


Y


# In[20]:


#Using Multinomial Naive Bayes with TF-IDF vectorizer
model = MultinomialNB()
model.fit(X_train, Y_train)


# In[21]:


ypred = model.predict(X_test)


# In[22]:


print("Accuracy Score for Naive Bayes Model using TF-IDF vectorizer is : ", accuracy_score(Y_test, ypred))


# In[23]:


from sklearn.metrics import classification_report

print("Classification Report for Naive Bayes Model using TF-IDF vectorizer : \n\n", classification_report(Y_test, ypred))


# In[24]:


count = CountVectorizer()
x = count.fit_transform(data['SentimentText'])


# In[25]:


y = data.iloc[:,[1]].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)


# In[26]:


#Using Multinomial Naive Bayes with CountVectorizer
m1 = MultinomialNB()
m1.fit(x_train, y_train)


# In[27]:


y2pred = m1.predict(x_test)


# In[28]:


print("Accuracy Score for Naive Bayes Model using CountVectorizer is : ", accuracy_score(y_test, y2pred))


# In[29]:


print("Classification Report for Naive Bayes model using CountVectorizer : \n\n", classification_report(y_test, y2pred))


# In[30]:


#Using Logistic Regression with TF-IDF vectorizer
lrm1 = LogisticRegression(solver='lbfgs')
lrm1.fit(X_train, Y_train)
lr1pred = lrm1.predict(X_test)
print("Accuracy Score for Logistic Regression Model (using TF-IDF vectorizer) is : ", accuracy_score(Y_test, lr1pred))


# In[31]:


print("Classification Report for Logistic Regression Model using TF-IDF vectorizer : \n\n", classification_report(Y_test, lr1pred))


# In[32]:


#Using Logistic Regression with CountVectorizer
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(x_train, y_train)


# In[33]:


lrpred = LR_model.predict(x_test)
print("Accuracy Score for Logistic Regression Model (using CountVectorizer) is : ", accuracy_score(y_test, lrpred))


# In[34]:


print("Classification Report for Logistic Regression Model using CountVectorizer : \n\n", classification_report(y_test, lrpred))

