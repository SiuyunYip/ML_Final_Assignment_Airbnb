import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def moveCol2Back(df):
    col_to_move = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                   'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                   'review_scores_value']

    col_map = {}
    for col in col_to_move:
        col_map[col] = np.array(df[col]).astype(float)
    df.drop(col_to_move, axis=1, inplace=True)
    for col in col_to_move:
        df[col] = col_map.get(col)


# binning review ratings
def binTarget(column, df):
    target_col = np.array(df[column]).astype(float)
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

    return target_col


# e.g.
# 0: 5.0 1637
# 1: [4.88, 5) 945
# 2: [4.73, 4.88) 1207
# 3: [4.5, 4.73) 1216
# 4: [0.0, 4.5) 1204

def readData(file_path, target_column):
    listings = pd.read_csv(file_path, index_col=0, sep=',')
    listings.drop(['description', 'name', 'comments'], axis=1, inplace=True)
    moveCol2Back(listings)

    train_cols = [listings.columns[0]]
    X = listings.iloc[:, 0]
    for i in range(1, len(listings.columns) - 7):
        train_cols.append(listings.columns[i])
        X2 = listings.iloc[:, i]
        X = np.column_stack((X, X2))

    y = binTarget(target_column, listings)
    return X, y, train_cols


train_cols = ['host_is_superhost', 'host_listings_count', 'host_identity_verified',
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
              'dist_to_attr1', 'dist_to_attr2', 'dist_to_attr3', 'owner', 'remark', 'landlord',
              'checkin', 'helpful', 'bakeri', 'thought', 'mountain', 'connolli', 'anywher', 'plenti',
              'simpl', 'bring', 'environ', 'polit', 'nice', 'actual', 'enough', 'ador', 'awar',
              'option', 'liffey', 'sparkl', 'cater', 'reach', 'towel', 'size']


def readSpecificCols(file_path, cols_to_read, target_columnn):
    listings = pd.read_csv(file_path, index_col=0, sep=',')

    X = np.array(listings[cols_to_read[0]])
    for i in range(1, len(cols_to_read)):
        X2 = np.array(listings[cols_to_read[i]])
        X = np.column_stack((X, X2))

    y = binTarget(target_columnn, listings)

    return X, y


def plotFeatureWeights(fig, data, plotArgs, title, cols_to_plot):
    feature_weights = abs(data)
    feature_weights = 100.0 * (feature_weights / feature_weights.max())
    sorted_idx = np.argsort(feature_weights)[0:20]
    pos = np.arange(sorted_idx.shape[0]) + .5

    ax = fig.add_subplot(plotArgs[0], plotArgs[1], plotArgs[2])
    ax.barh(pos, feature_weights[sorted_idx], align='center')
    ax.set_yticks(pos)
    y_labels = []
    for idx in sorted_idx:
        y_labels.append(cols_to_plot[idx])
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Relative Feature Weights')
    ax.set_title(title)


def plotWeightsOfTopFeatures(C, cols_to_plot, X_train, Y_train, X_test, Y_test):
    model = LogisticRegression(C=C, random_state=0, solver='newton-cg', multi_class='multinomial')
    model.fit(X_train, Y_train)
    print("LogisticRegression with C = ", C)
    print('Train accuracy score:', model.score(X_test, Y_test))
    print('Test accuracy score:', model.score(X_test, Y_test))

    fig = plt.figure(figsize=(16, 8))

    plotFeatureWeights(fig, model.coef_[0], (2, 3, 1), 'Class = 0', cols_to_plot)
    plotFeatureWeights(fig, model.coef_[1], (2, 3, 2), 'Class = 1', cols_to_plot)
    plotFeatureWeights(fig, model.coef_[2], (2, 3, 3), 'Class = 2', cols_to_plot)
    plotFeatureWeights(fig, model.coef_[3], (2, 3, 4), 'Class = 3', cols_to_plot)
    plotFeatureWeights(fig, model.coef_[3], (2, 3, 5), 'Class = 4', cols_to_plot)

    plt.tight_layout()
    plt.show()


# see the feature importance
X, y = readSpecificCols('data/targets/location_comment.csv', train_cols, 'review_scores_location')
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)
plotWeightsOfTopFeatures(0.05, train_cols, Xtrain, Ytrain, Xtest, Ytest)

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def getBestParamForKnn(K_range, X, y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

    param_grid = {"n_neighbors": K_range, "p": [1, 2]}
    model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5).fit(Xtrain, Ytrain)
    print('Best parameters: ', model.best_params_)
    Ypredict = model.best_estimator_.predict(Xtest)
    accuracy = np.count_nonzero((Ypredict == Ytest) == True) / len(Ytest)
    print("Prediction accuracy is:", accuracy)

    return model.best_estimator_


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


def plot_F1_Score(file_to_read, targets, p_range, model_name):
    plt.figure(figsize=(8, 5.5))
    for i in range(len(file_to_read)):
        X, y, train_cols = readData('data/targets/' + files_to_read[i], targets[i])
        mean_error, std_error = [], []
        cv = KFold(n_splits=5, shuffle=False)
        print('Start', targets[i])
        for p in p_range:
            tmp_mean, tmp_std = [], []
            for train_index, test_index in cv.split(X):
                X_train, X_test = X[train_index[0]:train_index[-1] + 1], X[test_index[0]:test_index[-1] + 1]
                y_train, y_test = y[train_index[0]:train_index[-1] + 1], y[test_index[0]:test_index[-1] + 1]

                if model_name == 'logistic':
                    model = LogisticRegression(C=p, random_state=0, solver='newton-cg', multi_class='multinomial').fit(
                        X_train, y_train)
                elif model_name == 'knn':
                    model = KNeighborsClassifier(n_neighbors=p, weights='uniform').fit(X_train, y_train)
                else:
                    break

                y_pred = model.predict(X_test)
                score = f1_score(y_test, y_pred, average=None)
                tmp_mean.append(np.array(score).mean())
                tmp_std.append(np.array(score).std())

            print('Mean f1_score: ', np.array(tmp_mean).mean())
            print('Standard f1_score: ', np.array(tmp_mean).std())
            print()
            mean_error.append(np.array(tmp_mean).mean())
            std_error.append(np.array(tmp_std).std())

        if model_name == 'knn':
            getBestParamForKnn(p_range, X, y)
        plt.errorbar(p_range, mean_error, yerr=std_error, linewidth=2, label=targets[i])
        print("=================")

    if model_name == 'knn':
        plt.gca().set(xlabel='K', ylabel="F1 score")
    else:
        plt.gca().set(xlabel='Ci', ylabel="F1 score")

    plt.legend(loc='upper right')
    plt.show()


files_to_read = ['cleanliness_comment.csv', 'accuracy_comment.csv', 'checkin_comment.csv',
                 'communication_comment.csv', 'location_comment.csv', 'value_comment.csv',
                 'ratings_comment.csv']
targets = ['review_scores_cleanliness', 'review_scores_accuracy', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location', 'review_scores_value',
           'review_scores_rating']

#
C = [0.001, 0.01, 0.05, 0.1, 1]
# mean_error_analysis(files_to_read, targets, C)
plot_F1_Score(files_to_read, targets, C, 'logistic')

# K_range = [5, 50, 100, 150, 200, 250, 300]
K_range = [1, 3, 5, 15, 25, 50, 100]
plot_F1_Score(files_to_read, targets, K_range, 'knn')

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def plot_Confusion_Matrix(model, Xtest, Ytest):
    predictions = model.predict(Xtest)
    cm = confusion_matrix(Ytest, predictions, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot()


from sklearn.dummy import DummyClassifier


def plotDummy(file_to_read):
    for i in range(len(file_to_read)):
        print('start', targets[i])
        X, y, _ = readData('data/targets/' + file_to_read[i], targets[i])
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

        dummy_model = DummyClassifier(strategy='most_frequent').fit(Xtrain, Ytrain)
        print('Scores of Dummy model: ', dummy_model.score(Xtest, Ytest))
        plot_Confusion_Matrix(dummy_model, Xtest, Ytest)
        print('=========================')


# plotDummy(files_to_read)


from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from numpy import interp
from itertools import cycle


def plotROCCurve(model_name, file_to_read, targets, n_classes):
    K_list = [25, 15, 15, 15, 15, 50, 15]
    C_list = [0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01]
    for j in range(len(file_to_read)):
        X, y, _ = readData('data/targets/' + file_to_read[j], targets[j])
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

        if model_name == 'LR':
            model = LogisticRegression(C=C_list[j], random_state=0, solver='newton-cg', multi_class='multinomial').fit(
                Xtrain, Ytrain)
        elif model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=K_list[j], weights='uniform').fit(Xtrain, Ytrain)
        else:
            model = DummyClassifier(strategy='most_frequent').fit(Xtrain, Ytrain)

        Ypredict = model.predict(Xtest)
        Ytest = Ytest.ravel()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(Ytest))[:, i],
                                          np.array(pd.get_dummies(Ypredict))[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr / n_classes
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        lw = 2
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='green', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of ' + model_name + ' (' + targets[j] + ')')
        plt.legend(loc="lower right")
        plt.show()


plotROCCurve('LR', files_to_read, targets, 5)
plotROCCurve('knn', files_to_read, targets, 5)
plotROCCurve('dummy', files_to_read, targets, 1)


def scoreModel(model_name, file_to_read, targets):
    K_list = [25, 15, 15, 15, 15, 50, 15]
    C_list = [0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01]
    for j in range(len(file_to_read)):
        X, y, _ = readData('data/targets/' + file_to_read[j], targets[j])
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

        if model_name == 'LR':
            model = LogisticRegression(C=C_list[j], random_state=0, solver='newton-cg', multi_class='multinomial').fit(
                Xtrain, Ytrain)
        elif model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=K_list[j], weights='uniform').fit(Xtrain, Ytrain)
        else:
            model = DummyClassifier(strategy='most_frequent').fit(Xtrain, Ytrain)

        print('start', targets[j], '(' + model_name + ')')
        print(model.score(Xtest, Ytest))
        print('======================')


scoreModel('LR', files_to_read, targets)
scoreModel('knn', files_to_read, targets)
