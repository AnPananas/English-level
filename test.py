


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pandas as pd
import re
import pysrt
from pathlib import Path

import sklearn
import codecs

from tqdm import tqdm
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    f1_score, roc_auc_score,
    classification_report, make_scorer
)
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
    train_test_split,
     GridSearchCV
)


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')

import pickle


# Сформируйте путь к файлу
file_path = 'D:/yandex_practicum/Projects/streamlit/model/gradient_boosting_model_Clf.pkl'

# Попробуйте загрузить модель
try:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    st.write("Модель предсказания успешно загружена!")
except FileNotFoundError:
    st.write("Файл модели  предсказания не найден.")



# Сформируйте путь к файлу
file_path_tf = 'D:/yandex_practicum/Projects/streamlit/model/tfidf_vectorizer.pkl'

# Попробуйте загрузить модель
try:
    with open(file_path_tf, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    st.write("Модель векторизатора успешно загружена!")
except FileNotFoundError:
    st.write("Файл модели векторизатора не найден.")


st.title('Определение сложности английского в фильме')
st.subheader('Привет, эта страница создана для того,\
              чтобы подсказать, соответсвует ли выбранный\
              Вами фильм нужному уровню английского.\
             Желаю успехов в освоении языка!')

# загрузкf файла
uploaded_file = st.file_uploader("Загрузите файл с субтитрами (.srt)", type=["srt"])

if uploaded_file is not None:
    # Прочитайте содержимое загруженного файла
    subtitles_text = uploaded_file.read().decode('iso-8859-1')
    subtitles_text = [subtitles_text]
    subtitles_text = pd.Series(subtitles_text)
    # разделяем по переходам на новую строку и заменяем на пробел
    subtitles_text = subtitles_text.apply(lambda x:' '.join(x.split('\n')))

    def extract_proper_nouns(text):
        # Токенизация слов
        words = nltk.word_tokenize(text)
        # Определение частей речи
        tagged_words = nltk.pos_tag(words)
        # Отбор имен собственных (NNP и NNPS)
        proper_nouns = [word for word, pos in tagged_words if pos not in ['NNP', 'NNPS']]

        return " ".join(proper_nouns)
    
    subtitles_text = subtitles_text.apply(extract_proper_nouns)

    subtitles_text = subtitles_text.str.lower()
    
    # Удаляем все символы кроме букв, апострофов(они нужны для распознавания некоторых стоп слов) и пробелов

    def remove_non_alphanumeric_except_apostrophe(input_string):
        # Используем регулярное выражение для удаления всех символов, кроме букв, пробелов и апострофа
        cleaned_string = re.sub(r"[^A-Za-z ']+", "", input_string)
        return cleaned_string
    
    subtitles_text = subtitles_text.apply(remove_non_alphanumeric_except_apostrophe)
    
    # для лемматизация нужно обозначить тэгами части речи

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # Проводим лемматизацию и токенизацию
    lemmatizer = nltk.stem.WordNetLemmatizer()
    def lemmatize_text(text):

        return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]

    subtitles_text = subtitles_text.apply(lemmatize_text)

    # Удаляем стоп слова

    def stop_words(text):
        stop_words = set(stopwords.words('english') + ['n']) # остались буквы n, добавим их к стоп словам
        return [word for word in text if not word in stop_words]

    subtitles_text = subtitles_text.apply(stop_words)

    # Используем регулярное выражение для удаления всех символов, кроме букв
    def remove_non_alphanumeric(input_string):

        input_string = " ".join(input_string)
        cleaned_string = re.sub(r"[^A-Za-z ]+", " ", input_string)
        return cleaned_string
    
    subtitles_text = subtitles_text.apply(remove_non_alphanumeric)

    subtitles_text = subtitles_text.apply(lemmatize_text)

    subtitles_text = subtitles_text.apply(stop_words)

    #преобразуем текст  в вектор методом  TF-IDF

    

    text =subtitles_text.apply(lambda x:  " ".join(x))
    values = tfidf_vectorizer.transform(text)


    #st.write(subtitles_text[:40])
    
    # Проведите классификацию с использованием обученной модели
    predicted_level = model.predict(values)[0]
   
    
    np.set_printoptions(precision=3, suppress=True)
    # Отобразите результат классификации
    st.write("Уровень английского в субтитрах:", predicted_level)
   

    predicted_level = model.predict(values)[0]
    proba_predictions = model.predict_proba(values)

    class_names =  ['A2', 'A2/A2+', 'A2/A2+, B1', 'B1','B1, B2','B2', 'C1', ]  # Замените на свои названия классов

    # Отобразите вероятности принадлежности уровню английского для каждого класса
    #st.write("Вероятности принадлежности уровеню английского в субтитрах:")
    #for class_label, probs in zip(class_names, proba_predictions[0]):
    #    st.write(f"Уровень {class_label}: {probs:.4f}")   
    proba_df = pd.DataFrame(proba_predictions, columns=class_names)

    # Отобразите вероятности принадлежности уровню английского с помощью seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=proba_df)
    plt.xlabel("Уровень английского")
    plt.ylabel("Вероятность")
    plt.title("Вероятности принадлежности уровню английского")

    # Добавление значений над столбцами
    for index, row in proba_df.iterrows():
        for i, value in enumerate(row):
            plt.text(i, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10)
            
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(plt)

# Загрузка датасета с названиями фильмов и уровнем английского
film_data = pd.read_excel("movies_labels.xlsx")

# Список уникальных уровней английского
english_levels = film_data["Level"].unique()

# Боковая панель с виджетом выбора уровня английского
selected_levels = st.sidebar.multiselect("Выберите уровни английского", english_levels)

# Отфильтрованный датасет на основе выбранных уровней
filtered_films = film_data[film_data["Level"].isin(selected_levels)]
# Добавляем текст с описанием на боковую панель
st.sidebar.write("Здесь вы можете выбрать уровень английского и мы предложим список \
                 подходящих фильмов.")
# Отображение списка фильмов на боковой панели
st.sidebar.write("Фильмы по выбранным уровням английского:")
for film_name in filtered_films["Movie"]:
    st.sidebar.write(film_name)

  