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
import matplotlib
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
pd.set_option('display.width', 300)
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
print(nltk.data.path)

###############################
# Lemmatization - Kelimeleri köklerine indirgeme işlemidir.
###############################
# nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################
# 2. Text Visualization
##################################################
###############################
# Terim Frekanslarının Hesaplanması
###############################
# Kelimelerin frekanslarını bulup görselleştirme işlemi yapalım. tf=turn_frekansy
tf = df['reviewText'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################
matplotlib.use('TkAgg')
fig, ax = plt.subplots()

# Verilerinizi filtreleyin
words = tf[tf["tf"] > 500]['words'].tolist()
tf_values = tf[tf["tf"] > 500]['tf'].tolist()

# Bar rengi ve etiket
bar_color = 'blue'

# Çubuk grafiği çizin
ax.bar(words, tf_values, color=bar_color)

# Grafiğin detaylarını ekleyin
ax.set_ylabel('TF Values')
ax.set_xlabel('Words')
ax.set_title('Word Frequency with TF > 500')

# Grafiği göster
plt.show()

###############################
# Wordcloud
###############################
# Satırları tek bir text dosyası gibi olması için birleştirme yapalım.
text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

###############################
# Şablonlara Göre Wordcloud
###############################

tr_mask = np. array(Image.open("tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10,10])
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()

##################################################
# 3. Sentiment Analysis (Duygu Analizi)
##################################################


###############################
# 4. Feature Engineering
###############################


###############################
# Count Vectors
###############################

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)


# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin numerik temsilleri

# ngram


###############################
# Count Vectors
###############################


# word frekans


# n-gram frekans


###############################
# TF-IDF
###############################


###############################
# 5. Sentiment Modeling
###############################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

###############################
# Logistic Regression
###############################


###############################
# Random Forests
###############################

# Count Vectors


# TF-IDF Word-Level


# TF-IDF N-GRAM


###############################
# Hiperparametre Optimizasyonu
###############################
