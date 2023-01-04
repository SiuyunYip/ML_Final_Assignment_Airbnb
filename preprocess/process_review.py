import re

import pandas as pd
import numpy as np
from googletrans import Translator
from langdetect import detect, LangDetectException

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')
# stop_words = nltk.corpus.stopwords.words('english')
#
# reviews = pd.read_csv('data/reviews.csv', sep=',')
# # drop null rows
# print(reviews.isnull().sum())
# reviews = reviews.dropna()
# print(reviews.shape)
#
#
# def convert2Eng(comments):
#     for i in range(len(comments)):
#         comments[i] = comments[i].replace("<br/>", "").strip()
#         try:
#             if detect(comments[i]) != 'en':
#                 print(detect(comments[i]))
#                 translator = Translator()
#                 res = translator.translate(comments[i])
#                 comments[i] = res
#         except LangDetectException as e:
#             print('Exception occurs: ', e)
#
#
# # comment_list = np.array(reviews['comments']).astype(str)
# # convert2Eng(comment_list)
# # reviews['translated_comments'] = comment_list
# # reviews.to_csv('./data/translated_reviews.csv', index=True)
#
# reviews = pd.read_csv('data/translated_reviews.csv', sep=',')
# reviews.dropna(subset=['listing_id', 'comments'], inplace=True)
# comment_list = np.array(reviews['comments']).astype(str)
# id_list = np.array(reviews['listing_id']).astype(str)
#
#
# def remove_punctuation(comment):
#     new_str = comment.replace(".", "")
#     new_str = new_str.replace(",", "")
#     new_str = new_str.replace("-", "")
#     new_str = new_str.replace("--", "")
#     new_str = new_str.replace("&", "")
#     new_str = new_str.replace("!", "")
#     new_str = new_str.replace(";", "")
#
#     return new_str
#
#
# comments = []
# for comment in comment_list:
#     # if not comment or type(comment) != str:
#     #     print(comment)
#     #     continue
#     comment = remove_punctuation(comment)
#
#     tokens = word_tokenize(comment)
#     # remove special character and number
#     # removed_tokens = [token for token in tokens if re.findall('^[a-z]+$', token)]
#     removed_tokens = [re.sub(r"[^A-Za-z]", " ", token.strip()).strip() for token in tokens]
#
#     lower_tokens = [token.lower() for token in removed_tokens]
#
#     stopremoved_token = [token for token in lower_tokens if token not in stop_words and len(token) > 4]
#
#     stemmer = PorterStemmer()
#     stem_tokens = [stemmer.stem(token) for token in stopremoved_token]
#
#     new_comment = ''
#     for token in stem_tokens:
#         new_comment = new_comment + token + ' '
#
#     comments.append(new_comment.strip())
#
#
# reviews['comments'] = comments

# # translate will cost a lot of time, comment out
# reviews.to_csv('./data/translated_reviews.csv', index=True)

# merge multiple reviews based on listing_id
reviews = pd.read_csv('data/translated_reviews.csv', sep=',')
comments = np.array(reviews['comments'])
id_list = np.array(reviews['listing_id'])
reviews.dropna()

id_comment_map = {}
for i in range(len(id_list)):
    if str(comments[i]) == 'nan':
        continue
    l_id = id_list[i]
    if l_id in id_comment_map:
        try:
            new_comment = id_comment_map.get(l_id)
            new_comment = new_comment + comments[i] + ' '
            id_comment_map[l_id] = new_comment.strip()
        except Exception as e:
            print(e)
    else:
        id_comment_map[l_id] = comments[i]

new_ids = []
new_comments = []
for entry in id_comment_map.items():
    new_ids.append(entry[0])
    new_comments.append(entry[1])

reviews['listing_id'] = pd.Series(new_ids)
reviews['comments'] = pd.Series(new_comments)

reviews.to_csv('./data/merge_reviews.csv', index=True)

# format the reviews file
reviews = pd.read_csv('../data/merge_reviews.csv', sep=',')


def reformat(reviews):
    reviews.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
    reviews.dropna(subset=['listing_id', 'comments'], inplace=True)
    reviews.to_csv('./data/merge_reviews.csv', index=False)
    print(reviews.shape)


# reformat(reviews)

# merge comments on listing_id and store into new csv
def merge_comment(reviews):
    listings = pd.read_csv('../data/processed_listings.csv', sep=',')
    listing_id = np.array(listings['id']).astype(str)

    review_map = {}
    for i in range(len(reviews['listing_id'])):
        review_map[str(reviews['listing_id'][i]).split('.')[0]] = reviews['comments'][i]

    comments = []
    for l_id in listing_id:
        c = review_map.get(l_id)
        if c:
            comments.append(c)
        else:
            comments.append('nan')

    listings['comments'] = comments
    listings.to_csv('data/merge_listings.csv', index=False)


# merge_comment(reviews)

listings = pd.read_csv('../data/merge_listings.csv', sep=',')


def generateBowBasedOnTarget(target, max_df=0.1, min_df=20):
    # select the comments with target of 5.0 to generate BOW
    comment_list = listings['comments'].values.astype('U')
    comments_scores_5 = []
    for i in range(len(comment_list)):
        if float(listings[target][i]) == 5.0:
            comments_scores_5.append(comment_list[i])

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(comments_scores_5)
    print(vectorizer.get_feature_names_out())
    print(X.toarray())

    return vectorizer.get_feature_names_out()


def oneHotEncode(comments, bow, save_csv):
    bow_map = {}
    # initiate the bow map to the format of token: list
    for b in bow:
        bow_map[b] = []

    for comment in comments:
        t_map = {}
        c_arr = comment.split(' ')
        for token in c_arr:
            if token in bow:
                if token in t_map:
                    t_map[token] = t_map.get(token) + 1
                else:
                    t_map[token] = 1

        for b in bow:
            freq_list = bow_map.get(b)
            if b in t_map:
                freq_list.append(t_map.get(b))
            else:
                freq_list.append(0)

            bow_map[b] = freq_list

    for entry in bow_map.items():
        listings[entry[0]] = entry[1]

    listings.to_csv('./data/' + save_csv, index=False)


comments = np.array(listings['comments']).astype(str)

bow_ratings = generateBowBasedOnTarget('review_scores_rating')
oneHotEncode(comments, bow_ratings, 'ratings_comment.csv')

bow_accuracy = generateBowBasedOnTarget('review_scores_accuracy')
oneHotEncode(comments, bow_accuracy, 'accuracy_comment.csv')

bow_cleanliness = generateBowBasedOnTarget('review_scores_cleanliness')
oneHotEncode(comments, bow_cleanliness, 'cleanliness_comment.csv')

bow_checkin = generateBowBasedOnTarget('review_scores_checkin')
oneHotEncode(comments, bow_checkin, 'checkin_comment.csv')

bow_communication = generateBowBasedOnTarget('review_scores_communication')
oneHotEncode(comments, bow_communication, 'communication_comment.csv')

bow_location = generateBowBasedOnTarget('review_scores_location')
oneHotEncode(comments, bow_location, 'location_comment.csv')

bow_value = generateBowBasedOnTarget('review_scores_value')
oneHotEncode(comments, bow_value, 'value_comment.csv')
