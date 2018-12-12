"""

Skeleton code. Note that this is not valid code due to all the dots.

"""
import json
import nltk
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


RANDOM = False
IS_TUNING = True
TARGET = 'ratingOverall'
FEATURE_COLS = ['ratingOverall', 'annotatorId', 'ratingRoom', 'hotelId', 'ratingLocation', 'ratingCleanliness', 'ratingService', 'ratingBusiness', 'ratingValue', 'ratingCheckin']


def create_feature_sets(data):
    # create feature sets
    if RANDOM:
        random.shuffle(data)

    feature_sets = [gen_features(item) for item in data]
    
    df = pd.DataFrame.from_dict(feature_sets, orient='columns', dtype='int')
    df[TARGET] = df[TARGET] - 1
    X = df.drop(columns=[TARGET], axis=1, inplace=False)
    y = df[TARGET]

    return train_test_split(X, y, train_size=0.75)


def gen_features(item):
    feature = {col: item[col] for col in FEATURE_COLS}

    # TODO extract features from review
    # review = item['segments']
    # ...
    # feature['new_feature1'] = ...
    # feature['new_feature2'] = ...
    # feature['new_feature3'] = ...

    return feature


def train_classifier(X_train, y_train):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 5,
        'num_leaves': 15,
        "num_threads": 4,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 500,
        'verbose': 0
    }

    # train
    classifier = lgb.train(params,
        lgb_train,
        num_boost_round=10,
        valid_sets=lgb_train,  # eval training data
        categorical_feature=[21])

    return classifier


def evaluate_classifier(classifier, X_test, y_test):
    # get the accuracy and print it
    y_pred = classifier.predict(X_test, num_iteration=classifier.best_iteration)
    y_pred_max = [np.argmax(pred) for pred in y_pred]
    print('Mean abs error is:', mean_absolute_error(y_test, y_pred_max))


def train_gridcv(X_train, y_train):
    # Create classifier to use. Note that parameters have to be input manually
    classifier = lgb.LGBMClassifier(boosting_type= 'gbdt', 
        objective = 'multiclass',
        n_jobs = 4,
        is_unbalance = True,
        max_depth = -1,
        max_bin = 512,
        verbose = 0)

    # Create parameters to search
    gridParams = {
        'learning_rate': [0.005, 0.01, 0.02],
        'n_estimators': [50, 100, 500],
        'num_leaves': [15, 31],
        'colsample_bytree' : [0.5, 1]
    }

    # Create the grid
    grid = GridSearchCV(classifier, gridParams, verbose=0, cv=3, n_jobs=1, scoring='neg_mean_absolute_error')
    
    # Run the grid
    print("Training GridSearchCV started...")
    grid.fit(X_train, y_train, eval_metric='multi_logloss')
    print("Training GridSearchCV ended...")

    save_grid_results(grid)
    

def save_grid_results(grid):
    # Print the best parameters found
    f = open("Tuning.txt", "w")
    f.write('Best params are: ' + str(grid.best_params_) + '\n')
    f.write('Best score is: ' + str(grid.best_score_) + '\n')
    f.close()


def load_data(filename):
    openfile = open(filename, 'r')
    data = [json.loads(line) for line in openfile.readlines()]
    openfile.close()
    return data


if __name__ == '__main__':

    # read corpus name from args
    filename = 'data.json'
    data = load_data(filename)

    # split train and test set
    X_train, X_test, y_train, y_test = create_feature_sets(data)
    
    if IS_TUNING:
        train_gridcv(X_train, y_train)
    else:
        # train classifier
        classifier = train_classifier(X_train, y_train)
        # evaluate
        evaluate_classifier(classifier, X_test, y_test)
    