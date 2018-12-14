import json
import nltk
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


IS_TUNING = False
LINE_SEPARATOR = '\n'

TARGET = 'overall_ratingsource'
FEATURES = ['doc_id', 'city', 'country', 'num_reviews', 'sentimental', 'cleaniness', 'room', 'service', 'location', 'value', 'food']


def create_feature_sets(df):
    # create feature sets
    
    df = gen_review_features(df)

    # for col in FEATURES:
    #     if col != TARGET:
    #         _df = df[df[col] >= 0]
    #         df[df[col] == -1] = _df[col].mean()

    X = df.drop(columns=[TARGET], axis=1, inplace=False)
    y = df[TARGET]

    return train_test_split(X, y, train_size=0.9, random_state=1)


def gen_review_features(df):
    # Extract features from review

    fileroot = 'data/reviews/'

    # Add or fix features below
    df['sentimental'] = 0
    df['cleaniness'] = 0
    df['room'] = 0
    df['service'] = 0
    df['location'] = 0
    df['value'] = 0
    df['food'] = 0

    # Iterate review docs
    for doc in df['doc_id']:
        filename = fileroot + doc

        openfile = open(filename,  'r', encoding='utf8', errors='ignore')
        raw = openfile.read() #.decode('utf8', 'ignore')
        reviews = raw.split(LINE_SEPARATOR)

        # Add or fix feature extraction functions below
        df.loc[df['doc_id'] == doc, 'sentimental'] = sentimental_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'cleaniness'] = cleaniness_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'room'] = room_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'service'] = service_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'location'] = location_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'value'] = value_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'food'] = food_from_review(reviews)

    return df


def sentimental_from_review(reviews):
    sentimental = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return sentimental


#This cleaniness would include information from all reviews of the hotel
def cleaniness_from_review(reviews):
    cleaniness = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    all_num_of_words = 0
    for review in reviews:
        num_of_words = len(word_tokenize(review))
        all_num_of_words += num_of_words
        cleaniness += is_clean(review) * num_of_words
    return cleaniness / all_num_of_words


def room_from_review(reviews):
    room = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return room


def service_from_review(reviews):
    service = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return service


def location_from_review(reviews):
    location = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return location


def value_from_review(reviews):
    value = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return value


def food_from_review(reviews):
    food = 0 # neutral: 0, negative: -1, positive: 1. Fractions are also allowed.
    for review in reviews:
        pass
    return food


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
        'learning_rate': 0.01,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.6,
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
    # for (a, b) in zip(y_test, y_pred_max):
    #     print(a, b)
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
    f = open("demo_tuning.txt", "w")
    f.write('Best params are: ' + str(grid.best_params_) + '\n')
    f.write('Best score is: ' + str(grid.best_score_) + '\n')
    f.close()


def load_data(filename):
    df = pd.read_csv(filename)
    return df


if __name__ == '__main__':

    # sample classifier on small data
    filename = 'data/hotels.csv'
    df = load_data(filename)

    # split train and test set
    X_train, X_test, y_train, y_test = create_feature_sets(df)
    
    # if IS_TUNING:
    #     train_gridcv(X_train, y_train)
    # else:
    #     # train classifier
    #     classifier = train_classifier(X_train, y_train)
    #     # evaluate
    #     evaluate_classifier(classifier, X_test, y_test)
