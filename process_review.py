import re

import pandas as pd
import numpy as np

# https://www.thepythoncode.com/article/translate-text-in-python
# import translator as translator
# import translators as ts
# from googletrans import Translator

import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import PorterStemmer

# translator = Translator()
nltk.download('punkt')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['</br>'])

reviews = pd.read_csv('data/reviews.csv', index_col=0, sep=',')
# drop null rows
print(reviews.isnull().sum())
reviews = reviews.dropna()

comment_list = np.array(reviews['comments']).astype(str)


def remove_punctuation(comment):
    new_str = comment.replace(".", "")
    new_str = new_str.replace(",", "")
    new_str = new_str.replace("-", "")
    new_str = new_str.replace("--", "")
    new_str = new_str.replace("&", "")
    new_str = new_str.replace("!", "")
    new_str = new_str.replace(";", "")

    return new_str


comments = []
for i in comment_list:
    if not i or type(i) != str:
        continue

    i = i.replace("<br/>", "").strip()
    i = remove_punctuation(i)
    # i = translator.translate(i)
    # i = ts.translate_text(i, to_language='en')
    # i = remove_punctuation(i.text)

    tokens = word_tokenize(i)
    # remove special character and number
    removed_tokens = [token for token in tokens if re.findall('^[a-z]+$', token)]

    lower_tokens = [token.lower() for token in removed_tokens]

    stemmer = PorterStemmer()
    stem_tokens = [stemmer.stem(token) for token in lower_tokens]

    stopremoved_token = [token for token in stem_tokens if token not in stop_words and len(token) > 4]

    comments.append(stopremoved_token)

print()
