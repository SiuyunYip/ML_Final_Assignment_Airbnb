import math
import operator
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

listings = pd.read_csv('data/listings.csv', index_col=0, sep=',')

# plot the distributions of each rating
ratings_to_plot = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                   'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                   'review_scores_value']
for rating in ratings_to_plot:
    # Plot Rating Scores Distribution Plot
    hist_kws = {"alpha": 0.3}
    plt.figure(figsize=(10, 8))
    plt.xticks(np.arange(0, 6, step=1))
    sns.distplot(listings[rating], hist_kws=hist_kws)
    # plt.show()

print(listings.shape)
print(listings.head())
listings.drop(['host_location', 'picture_url', 'host_url', 'host_name', 'host_id', 'host_thumbnail_url',
               'host_picture_url', 'host_has_profile_pic', 'listing_url', 'scrape_id', 'last_scraped',
               'source', 'calendar_last_scraped', 'calendar_updated', 'license', 'bathrooms',
               'neighbourhood_group_cleansed'],
              axis=1, inplace=True)
print(listings.shape)


def checkNilFeature(data, threshold=0):
    cols_have_nil = []
    nil_nums = []
    for col in data.columns.to_list():
        if data[col].isna().sum() > threshold:
            cols_have_nil.append(col)
            nil_nums.append(listings[col].isnull().sum())

    plt.bar(cols_have_nil, nil_nums)
    plt.xticks(cols_have_nil, rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.title('Nil Values (listings Data)')
    plt.ylabel('Numbers of Nil Values')
    plt.tight_layout()
    # plt.show()


# checkNilFeature()
# checkNilFeature(1000)

# drop sample with empty first_review value
listings.dropna(subset=['first_review', 'review_scores_rating'], inplace=True)

cols_to_drop = ['neighborhood_overview', 'host_about', 'neighbourhood', 'first_review', 'host_since',
                'last_review', 'host_neighbourhood', 'neighbourhood_cleansed']
listings.drop(cols_to_drop, axis=1, inplace=True)
print(listings.shape)


# (6209, 56)
# checkNilFeature(listings)


def checkType(column_name):
    print('check types of column {}'.format(column_name))
    value_type = {}
    for row in listings[column_name]:
        if row in value_type:
            cnt = value_type[row]
            cnt += 1
            value_type[row] = cnt
        else:
            value_type[row] = 1

    print(value_type)


# check host response time types
checkType('host_response_time')


def processRspTime():
    import random

    # preprocess host response time
    rsp_time_list = np.array(listings.host_response_time).astype(str)
    for i in range(len(rsp_time_list)):
        if rsp_time_list[i] == 'within an hour':
            rsp_time_list[i] = 0.5
        elif rsp_time_list[i] == 'within a few hours':
            rsp_time_list[i] = 1.2
        elif rsp_time_list[i] == 'within a day':
            rsp_time_list[i] = 1.5
        elif rsp_time_list[i] == 'a few days or more':
            rsp_time_list[i] = 3
        else:
            rand_int = random.randint(0, 100)
            if rand_int < 68:
                rsp_time_list[i] = 0.5
            elif rand_int >= 68 and rand_int < 87:
                rsp_time_list[i] = 1.2
            elif rand_int >= 87 and rand_int < 98:
                rsp_time_list[i] = 1.5
            else:
                rsp_time_list[i] = 3

    listings.host_response_time = rsp_time_list


def processRspTimeV2():
    listings.host_response_time.fillna("unknown", inplace=True)
    print(listings.host_response_time.value_counts(normalize=True))
    dummy_resp_time = pd.get_dummies(listings.host_response_time, prefix='host_response_time')
    for col in dummy_resp_time.columns:
        listings[col] = dummy_resp_time[col]
    listings.drop('host_response_time', axis=1, inplace=True)



processRspTimeV2()

# check response rate type
checkType('host_response_rate')


def calMeanAndAssignVal(rate_column):
    cnt = 0
    sum_rate = 0.0
    for rate in rate_column:
        if type(rate) == str and rate != 'nan' and rate != 't':
            sum_rate += float(rate)
            cnt += 1

    avg_val = sum_rate / cnt
    for i in range(len(rate_column)):
        if rate_column[i] == 'nan' or rate_column[i] == 't':
            rate_column[i] = avg_val
        else:
            rate_column[i] = float(rate_column[i])

    return ["{:.2f}".format(rate / 100.0) for rate in rate_column]


def processRspAndAcceptRate(column):
    listings[column].fillna(-1, inplace=True)
    rate_column = listings[column].copy().astype(str)
    rate_column = [rate.replace('%', '', 1) for rate in rate_column]
    for i in range(len(rate_column)):
        if 90 <= int(rate_column[i]) <= 99:
            rate_column[i] = '90-99'
        elif 80 <= int(rate_column[i]) <= 89:
            rate_column[i] = '80-89'
        elif int(rate_column[i]) < 80:
            rate_column[i] = '0-79'

    listings[column] = rate_column
    dummy_rate = pd.get_dummies(listings[column], prefix=column)
    for col in dummy_rate:
        listings[col] = dummy_rate[col]

    listings.drop(column, axis=1, inplace=True)


processRspAndAcceptRate('host_response_rate')
processRspAndAcceptRate('host_acceptance_rate')


# interval: 100%, [90-99], [0-89]
# def processRspTimeRate():
#     time_rate = listings.host_response_rate.copy().astype(str)
#     time_rate = [rate.replace('%', '', 1) for rate in time_rate]
#
#     listings['host_response_rate'] = calMeanAndAssignVal(time_rate)
#

# interval: 100%, [90-99], [0-89]
# def processRspAcceptanceRate():
#     accpt_rate = listings.host_acceptance_rate.copy().astype(str)
#     accpt_rate = [accpt.replace('%', '', 1) for accpt in accpt_rate]
#
#     listings['host_acceptance_rate'] = calMeanAndAssignVal(accpt_rate)
#
#
# print('======================')
# print(listings.host_acceptance_rate.value_counts())
# print(pd.isnull(listings.host_acceptance_rate).sum())
# processRspAcceptanceRate()

def checkBoolCol(col_list):
    for col in col_list:
        print('Nan values of {}:'.format(col), pd.isna(listings[col]).sum())
        for i in listings[col]:
            if i != 't' and i != 'f':
                print(i)


columns_to_process = ['host_is_superhost', 'host_identity_verified',
                      'has_availability', 'instant_bookable']
checkBoolCol(columns_to_process)


def processBoolVal(bool_columns):
    for col in bool_columns:
        col_to_process = np.array(listings[col]).astype(str)
        for i in range(len(col_to_process)):
            if col_to_process[i] == 't':
                col_to_process[i] = 1
            elif col_to_process[i] == 'f':
                col_to_process[i] = 0
            else:
                col_to_process[i] = 0

        listings[col] = col_to_process


processBoolVal(columns_to_process)

# process host verifications
checkType('host_verifications')


def processVerification():
    nums_verification = []
    verifications = np.array(listings.host_verifications).astype(str)
    for i in range(len(verifications)):
        if verifications[i] == '[]':
            nums_verification.append(0)
        else:
            nums_verification.append(verifications[i].count(',') + 1)

    listings['nums_verification'] = nums_verification

    listings.drop('host_verifications', axis=1, inplace=True)


processVerification()

# Process bathroom text and generate two new columns
# recording number of bathroom and bathroom per person prespectively
checkType('bathrooms_text')


def processBathTxt():
    nums_bathroom = []
    bathroom_pp = []

    bathrooms = np.array(listings.bathrooms_text).astype(str)
    accommodates = np.array(listings.accommodates).astype(float)
    for i in range(len(bathrooms)):
        if bathrooms[i] == 'nan':
            nums_bathroom.append(0.0)
            bathroom_pp.append(0.00)
        elif bathrooms[i] in ['Shared half-bath', 'Half-bath', 'Private half-bath']:
            nums_bathroom.append(0.5)
            bathroom_pp.append('{:.2f}'.format(0.5 / accommodates[i]))
        else:
            num = '{:.1f}'.format(float(bathrooms[i].split(' ')[0]))
            nums_bathroom.append(num)
            bathroom_pp.append('{:.2f}'.format(float(num) / accommodates[i]))

    listings['nums_bathroom'] = nums_bathroom
    listings['bathroom_per_person'] = bathroom_pp

    listings.drop('bathrooms_text', axis=1, inplace=True)


processBathTxt()


# process bedrooms and beds
# fill empty value
def processBed():
    bedrooms = np.array(listings.bedrooms).astype(int)
    beds = np.array(listings.beds).astype(int)
    accommodates = np.array(listings.accommodates).astype(int)

    for i in range(len(accommodates)):
        if bedrooms[i] < 0:
            if accommodates[i] == 0:
                bedrooms[i] = 0
            elif accommodates[i] <= 3:
                bedrooms[i] = 1
            elif accommodates[i] <= 5:
                bedrooms[i] = 2
            else:
                bedrooms[i] = 3

        if beds[i] < 0:
            if accommodates[i] == 0:
                beds[i] = 0
            if accommodates[i] <= 2:
                beds[i] = 1
            elif accommodates[i] <= 4:
                beds[i] = 2
            else:
                beds[i] = 3

    listings.bedrooms = bedrooms
    listings.beds = beds


processBed()


# process price tag
def processPrice():
    prices = np.array(listings.price).astype(str)
    prices = [p.replace('$', '', 1) for p in prices]
    prices = [p.replace(',', '', 1) for p in prices]
    prices = [p.split('.')[0] for p in prices]
    # prices = ['{:.2f}'.format(float(int(p) / 100.0)) for p in prices]
    listings.price = prices


processPrice()


# process amenities
def processAmenities():
    amenity_map = {}

    amenities = np.array(listings.amenities).astype(str)
    for amenity in amenities:
        amenity = amenity.replace('[', '').replace(']', '')
        amenity_arr = amenity.split(',')
        amenity_arr = [re.sub(r"[^A-Za-z]", " ", a.strip()).strip() for a in amenity_arr]
        for a in amenity_arr:
            if a in amenity_map:
                cnt = amenity_map[a]
                cnt += 1
                amenity_map[a] = cnt
            else:
                amenity_map[a] = 1

    sorted_map = dict(sorted(amenity_map.items(), key=operator.itemgetter(1), reverse=True))

    # select amenities those appear in between 500 to 2000 times
    amenity_as_feature = []
    for item in sorted_map.items():
        if 500 <= item[1] <= 2000:
            amenity_as_feature.append(item[0])

    unnecessary = ['Lockbox', 'Private entrance', 'Paid parking off premises',
                   'Wine glasses', 'Laundromat nearby', 'TV with standard cable',
                   'Room darkening shades', 'Private patio or balcony', 'Paid parking off premises']

    # chosen amenities as new features inserted into the table
    amenity_as_feature = [a for a in amenity_as_feature if a not in unnecessary]
    for new_feature in amenity_as_feature:
        new_column = []
        for amenity in amenities:
            amenity = amenity.replace('[', '').replace(']', '')
            amenity_arr = amenity.split(',')
            amenity_arr = [re.sub(r"[^A-Za-z]", " ", a.strip()).strip() for a in amenity_arr]
            if new_feature in amenity_arr:
                new_column.append(1)
            else:
                new_column.append(0)

        listings[new_feature] = new_column

    listings.drop('amenities', axis=1, inplace=True)


processAmenities()


# calculate three nearest attractions around a listing using latitude and longitude
# quote from: https://stackoverflow.com/questions/3694380/calculating-distance-between-two-points-using-latitude-longitude
# convert decimal degrees to radians
def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / math.pi


def calDistance(lat1, lat2, lon1, lon2):
    theta = lon1 - lon2
    dist = math.sin(deg2rad(lat1)) * math.sin(deg2rad(lat2)) \
           + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.cos(deg2rad(theta))
    dist = math.acos(dist)
    dist = rad2deg(dist)
    dist = dist * 60 * 1.1515

    return dist * 1.609344


def featureDistances():
    # data from: https://data.gov.ie/dataset/attractions/resource/f6ccaa7b-6c59-4ea8-9da4-fd2fc3982872
    attractions = pd.read_csv('data/attractions.csv', index_col=0, sep=',')
    attr_long = np.array(attractions.Longitude).astype(float)
    attr_lat = np.array(attractions.Latitude).astype(float)
    longs = np.array(listings.longitude).astype(float)
    lats = np.array(listings.latitude).astype(float)

    dist_to_attr1 = []
    dist_to_attr2 = []
    dist_to_attr3 = []
    for i in range(len(longs)):
        distances = []
        for j in range(len(attr_long)):
            distances.append(calDistance(lats[i], attr_lat[j], longs[i], attr_long[j]))
        distances.sort()
        dist_to_attr1.append(distances[0])
        dist_to_attr2.append(distances[1])
        dist_to_attr3.append(distances[2])

    listings['dist_to_attr1'] = dist_to_attr1
    listings['dist_to_attr2'] = dist_to_attr2
    listings['dist_to_attr3'] = dist_to_attr3

    listings.drop(['latitude', 'longitude'], axis=1, inplace=True)


featureDistances()


# process property type
def processProperty():
    print(listings.property_type.value_counts())
    listings.property_type.replace({
        'Room in hotel': 'Entire',
        'Room in boutique hotel': 'Entire',
        'Room in hostel': 'Share',
        'Room in aparthotel': 'Entire',
        'Tiny home': 'Entire',
        'Room in serviced apartment': 'Entire',
    }, inplace=True)

    properties = np.array(listings.property_type).astype(str)
    for i in range(len(properties)):
        if properties[i].split(' ')[0] in ['Private', 'Shared']:
            properties[i] = 0
        elif properties[i].split(' ')[0] == 'Entire':
            properties[i] = 1
        else:
            properties[i] = 2

    listings.property_type = properties

    # drop room type because these two columns are highly correlated
    listings.drop('room_type', axis=1, inplace=True)


processProperty()

# process minimum+/maximum+ nights () which are highly correlated
print(sum((listings.minimum_nights == listings.minimum_minimum_nights) == False))  # 542
print(sum((listings.maximum_nights == listings.maximum_maximum_nights) == False))  # 1007
listings.drop(['minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
               'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'],
              axis=1, inplace=True)

# process availability column
print(listings.has_availability.value_counts())  # t: 7537 f: 29
print(pd.isnull(listings.availability_60).sum())
listings.drop(['has_availability', 'availability_30', 'availability_90', 'availability_365'],
              axis=1, inplace=True)

# process number of reviews
listings.drop(['number_of_reviews_ltm', 'number_of_reviews_l30d'], axis=1, inplace=True)

# process hot listings cnt
listings.drop(['host_total_listings_count'], axis=1, inplace=True)

# drop calculated_* columns
listings.drop(['calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
               'calculated_host_listings_count_shared_rooms'], axis=1, inplace=True)


def standardize(cols):
    for col in cols:
        listings[col].replace({
            '0.0': '0.1',
            '0': '0.1',
            0: 0.1,
            0.0: 0.1
        }, inplace=True)

        log_column = np.array(listings[col]).astype(float)
        log_column = [math.log(row) for row in log_column]

        listings[col] = log_column


cols_to_log = ['price', 'minimum_nights', 'maximum_nights', 'availability_60',
               'number_of_reviews', 'calculated_host_listings_count']
standardize(cols_to_log)


print(listings.shape)
listings.to_csv('./data/processed_listings.csv', index=True)
