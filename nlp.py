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

df["reviewText"].head()
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()  # metinlerdeki duygusal analizleri yapan bir method
sia.polarity_scores('The film was awesome')
# {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}

sia.polarity_scores("I like this music but it is not good as the other one")
# {'neg': 0.209, 'neu': 0.673, 'pos': 0.118, 'compound': -0.3311}

# Her bir satırdaki veriyi skorlayalım
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])

###############################
# 4. Feature Engineering
###############################

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]
###############################
# Count Vectors - Kelimelerin Vektörleştirilmesi
# Amaç : Kelimeleri sayısal temsillere dönüştürme işlemi
###############################

# Kelime Vektörleştirme Yöntemleri
# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)


# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin numerik temsilleri

# ngram :kelime öbeklerine göre özellik üretmeyi ifade eder.
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyonlarıını gösterir ve feature üretmek için kuallanılır"""
TextBlob(a).ngrams(3)

###############################
# Count Vectors
###############################
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()  # Eşsiz kelimelerin isimlerini getirelim

# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third','this'] bu her bir kelimeyi sutun olarak düşünüp. Cümlelerde geçiyor ise 1 geçmiyorsa 0
X_c.toarray()  # Kelimeleri nümerik olarak ifade ettik

# n-gram frekans
# Kelimeleri ikişerli(ngram_range=(2, 2)) öbekler halinde vektörleştirelim
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
X_n.toarray()

# Veri setimizdeki reviewText değerlerimizin count vector yöntemi ile vektörleştirelim.
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

vectorizer.get_feature_names_out()[10:15]
X_count.toarray()
###############################
# TF-IDF
###############################
# '''TF-IDF (Term Frequency-Inverse Document Frequency), bir kelimenin bir belge içindeki önemini değerlendiren bir istatistiksel ölçüttür. TF (Term Frequency), kelimenin belgede ne sıklıkta geçtiğini, IDF (Inverse Document Frequency) ise kelimenin tüm belgeler içinde ne kadar nadir olduğunu hesaplar. Bu kombinasyon, özellikle bilgi alma ve metin madenciliği uygulamalarında önemlidir.'''

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()  # ön tanımlı değeri word- kelime olarak çalışmaktadır.
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)


###############################
# 5. Sentiment Modeling
###############################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

###############################
# Logistic Regression - Amacımız gelen yorumun pozitif mi negatif mi olduğunu tahmin edecek bir model kurmak.
###############################
log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

# ÇIKTI : np.float64(0.830111902339776) sonuca göre tahminlerimizin %83 başarılı olacaktır..

new_review = pd.Series("this product is great")
new_review1 = pd.Series("look at that shit very bad")

# Gelen yorumu modele uygun formatın haline getirmek için tf-idf ugulayalım.
new_review = TfidfVectorizer().fit(X).transform(new_review)
new_review1 = TfidfVectorizer().fit(X).transform(new_review1)

log_model.predict(new_review)
log_model.predict(new_review1)

random_review = pd.Series(df['reviewText'].sample(1).values)

new_review = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(new_review)

###############################
# Random Forests
###############################

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model,
                X_count,
                y,
                cv=5,
                n_jobs=1).mean()  # ÇIKTI : np.float64(0.8419125127161751)

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model,
                X_count,
                y,
                cv=5,
                n_jobs=1).mean()  # ÇIKTI : np.float64(0.8443540183112919)


# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model,
                X_count,
                y,
                cv=5,
                n_jobs=1).mean()  # ÇIKTI :


###############################
# Hiperparametre Optimizasyonu
###############################
