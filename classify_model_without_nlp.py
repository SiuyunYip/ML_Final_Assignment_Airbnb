import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

listings = pd.read_csv('./data/processed_listings.csv', index_col=0, sep=',')
listings.drop(['description', 'name'], axis=1, inplace=True)

# binning review ratings
print(listings.shape)
cnt_1 = 0
cnt_2 = 0
cnt_3 = 0
cnt_4 = 0
cnt_5 = 0
for scores in listings.review_scores_rating:
    if scores == 5:
        cnt_1 += 1
    elif 4.88 <= scores < 5:
        cnt_2 += 1
    elif 4.73 <= scores < 4.88:
        cnt_3 += 1
    elif 4.5 <= scores < 4.73:
        cnt_4 += 1
    else:
        cnt_5 += 1

target_col = np.array(listings.review_scores_rating).astype(float)
for i in range(len(target_col)):
    if target_col[i] == 5:
        target_col[i] = 0
    elif 4.88 <= target_col[i] < 5:
        target_col[i] = 1
    elif 4.73 <= target_col[i] < 4.88:
        target_col[i] = 2
    elif 4.5 <= target_col[i] < 4.73:
        target_col[i] = 3
    else:
        target_col[i] = 4

# listings_classification = listings.assign(
#     rating_bins=pd.qcut(
#         listings['review_scores_rating'],
#         q=4,
#         duplicates='drop',
#         # labels=[0, 1, 2, 3]
#     )
# )
# 3    2453 (4.88, 5.0]
# 2    1263 (4.73, 4.88]
# 1    1068 (4.5, 4.73]
# 0    1425 (-0.001, 4.5]

# print(listings_classification['rating_bins'].value_counts())
# print(listings_classification.columns)
# 'host_is_superhost'
train_cols = ['host_listings_count', 'host_identity_verified',
              'property_type', 'accommodates', 'bedrooms', 'beds', 'price',
              'minimum_nights', 'maximum_nights', 'availability_60',
              'number_of_reviews', 'instant_bookable',
              'calculated_host_listings_count', 'reviews_per_month',
              'host_response_time_a few days or more', 'host_response_time_unknown',
              'host_response_time_within a day',
              'host_response_time_within a few hours',
              'host_response_time_within an hour', 'host_response_rate_0-79',
              'host_response_rate_100', 'host_response_rate_80-89',
              'host_response_rate_90-99', 'host_acceptance_rate_0-79',
              'host_acceptance_rate_100', 'host_acceptance_rate_80-89',
              'host_acceptance_rate_90-99', 'nums_verification', 'nums_bathroom',
              'bathroom_per_person', 'Coffee maker', 'Stove', 'Dishwasher',
              'Extra pillows and blankets', 'Lock on bedroom door',
              'Indoor fireplace', 'Free street parking', 'Dedicated workspace',
              'Elevator', 'Luggage dropoff allowed', 'Host greets you',
              'Hot water kettle', 'Shower gel', 'Cleaning products', 'Cable TV',
              'Bathtub', 'Toaster', 'Breakfast', 'Backyard', 'Freezer',
              'Dining table', 'Body soap', 'Patio or balcony',
              'Drying rack for clothing', 'Central heating', 'Outdoor dining area',
              'Outdoor furniture', 'Security cameras on property', 'Conditioner',
              'dist_to_attr1', 'dist_to_attr2', 'dist_to_attr3']

# target_col = np.array(listings_classification['rating_bins'])
X = np.array(listings.host_is_superhost)
for col in train_cols:
    X2 = np.array(listings[col])
    X = np.column_stack((X, X2))

train_cols.insert(0, 'host_is_superhost')


def plotFeatureWeights(data, plotArgs, title):
    feature_weights = abs(data)
    feature_weights = 100.0 * (feature_weights / feature_weights.max())
    sorted_idx = np.argsort(feature_weights)[0:10]
    pos = np.arange(sorted_idx.shape[0]) + .5
    ax = fig.add_subplot(plotArgs[0], plotArgs[1], plotArgs[2])
    ax.barh(pos, feature_weights[sorted_idx], align='center')
    ax.set_yticks(pos)
    y_labels = []
    for idx in sorted_idx:
        y_labels.append(train_cols[idx])
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Relative Feature Weights')
    ax.set_title(title)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, target_col, test_size=0.2)
C = [0.01, 0.1, 1, 10]
for i in C:
    model = LogisticRegression(C=i, random_state=0, solver='newton-cg', multi_class='multinomial')
    model.fit(Xtrain, Ytrain)
    print("LogisticRegression with C = ", i)
    print('Train accuracy score:', model.score(Xtrain, Ytrain))
    print('Test accuracy score:', model.score(Xtest, Ytest))

    fig = plt.figure(figsize=(16, 8))

    plotFeatureWeights(model.coef_[0], (2, 2, 1), 'Class = 0 with C = {}'.format(i))
    plotFeatureWeights(model.coef_[1], (2, 2, 2), 'Class = 1 with C = {}'.format(i))
    plotFeatureWeights(model.coef_[2], (2, 2, 3), 'Class = 2 with C = {}'.format(i))
    plotFeatureWeights(model.coef_[3], (2, 2, 4), 'Class = 3 with C = {}'.format(i))


    plt.tight_layout()
    plt.show()
