##################################################
# Introduction to Text Mining and Natural Language Processing
##################################################

##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing - Ölçüm değeri bir ifade etmeyen nokta,sayılar vs. kurtulma işlemi.
##################################################
df = pd.read_csv("datasets/amazon_reviews.csv", sep=",")
df.head()

###############################
# Normalizing Case Folding
###############################
df["reviewText"] = df["reviewText"].str.lower()

###############################
# Punctuations
###############################
# Verinin belirli bozukluklardan arındırılması.
df["reviewText"] = df["reviewText"].str.replace(r'[^\w\s]', '', regex=True)
# regular expression

###############################
# Numbers
###############################
df["reviewText"] = df["reviewText"].str.replace(r'\d', '', regex=True)

###############################
# Stopwords - dilde yaygın kullanılan ve bir anlam ifade etmeyen kelimelerin silinmesi
###############################
# nltk.download('stopwords')
sw = stopwords.words('english')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################
# Kelimelerin frekanslarını bulalaım
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df <= 1]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

###############################
# Tokenization - Cümleleri parçalamak
###############################
# nltk.download("punkt")
# TextBlob(x).words, metni kelimelere ayırarak her bir kelimeyi bir liste elemanı olarak döndüren bir işlemdir. Bu, metin analizinde veya metinden belirli özellikleri çıkarmada yaygın olarak kullanılır.
df['reviewText'].apply(lambda x: TextBlob(x).words).head()

###############################
# Lemmatization - Kelimeleri köklerine indirgeme işlemidir.
###############################
# nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

