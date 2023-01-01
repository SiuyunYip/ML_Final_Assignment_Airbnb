import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

listings = pd.read_csv('./data/processed_listings.csv', index_col=0, sep=',')
listings.drop(['description', 'name'], axis=1, inplace=True)

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

target_col = np.array(listings['review_scores_rating'])
X = np.array(listings.host_is_superhost)
for col in train_cols:
    X2 = np.array(listings[col])
    X = np.column_stack((X, X2))
train_cols.insert(0, 'host_is_superhost')

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, target_col, test_size=0.2)


from sklearn.linear_model import LinearRegression


model = LinearRegression().fit(Xtrain, Ytrain)

fig = plt.figure(figsize=(16, 8))

feature_weights = abs(model.coef_)
feature_weights = 100.0 * (feature_weights / feature_weights.max())
sorted_idx = np.argsort(feature_weights)[0:10]
pos = np.arange(sorted_idx.shape[0]) + .5
sorted_idx_list = list(sorted_idx)
ax = fig.add_subplot(1, 3, 1)
ax.barh(pos, feature_weights[sorted_idx], align='center')
ax.set_yticks(pos)
y_labels = []
for idx in sorted_idx:
    y_labels.append(train_cols[idx])
ax.set_yticklabels(y_labels, fontsize=8)
ax.set_xlabel('Relative Feature Importance')

plt.tight_layout()
plt.show()

print('Train accuracy score:', model.score(Xtrain, Ytrain))
print('Test accuracy score:', model.score(Xtest, Ytest))
