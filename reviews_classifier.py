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
# FEATURES = ['city', 'country', 'num_reviews']
FEATURES = ['doc_id', 'city', 'country', 'num_reviews', 'sentimental', 'cleaniness', 'room', 'service', 'location', 'value', 'food']


def create_feature_sets(df):
    # Create feature sets

    # Remove the hotels with num_of_reviews < 0
    df = df[df['num_reviews'] >= 0]

    # Generate features
    df = gen_review_features(df)

    # Encode str type
    for col in FEATURES:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes

    # for col in FEATURES:
    #     if col != TARGET:
    #         _df = df[df[col] >= 0]
    #         df[df[col] == -1] = _df[col].mean()

    X = df[FEATURES]
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
        reviews = openfile.read().split(LINE_SEPARATOR)
        openfile.close()

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

def quality(reviews):
    cleaniness_list = []
    room_list = []
    service_list = []
    location_list = []
    value_list = []
    food_list = []
    all_num_of_words = 0
    for review in reviews:
        num_of_words = len(word_tokenize(review))
        all_num_of_words += num_of_words
        cleaniness_list.append(is_clean(review) * num_of_words)
        room_list.append(nice_room(review) * num_of_words)
        service_list.append(nice_service(review) * num_of_words)
        location_list.append(nice_location(review) * num_of_words)
        value_list.append(nice_value(review) * num_of_words)
        food_list.append(nice_food(review) * num_of_words)
    return (to_quality_pair(cleaniness_list, all_num_of_words), \
        to_quality_pair(room_list, all_num_of_words),\
        to_quality_pair(service_list, all_num_of_words),\
        to_quality_pair(location_list, all_num_of_words),\
        to_quality_pair(value_list, all_num_of_words),\
        to_quality_pair(food_list, all_num_of_words)
        )

def to_quality_pair(quality_lsit, normalize_val):
    quality = np.array(quality_lsit) / normalize_val
    return (np.mean(quality), np.var(quality))

def is_clean(review):
    return 0

def nice_room(review):
    return 0

def nice_service(review):
    return 0

def nice_location(review):
    return 0

def nice_value(review):
    return 0

def nice_food(review):
    return 0


def train_classifier(X_train, y_train):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mean_absolute_error',
        'num_leaves': 7,
        "num_threads": 4,
        'learning_rate': 0.005,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'n_estimators': 1000,
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
    # Evaluate our classifier and print it

    y_pred = classifier.predict(X_test, num_iteration=classifier.best_iteration)

    # Post processing
    # X_num_reviews = X_test['num_reviews'].values
    # for i in range(len(X_num_reviews)):
    #     if X_num_reviews[i] < 0: y_pred[i] = -1

    print('Mean absolute error is:', mean_absolute_error(y_test, y_pred))


def train_gridcv(X_train, y_train):
    # Create classifier to use. Note that parameters have to be input manually
    classifier = lgb.LGBMRegressor(boosting_type= 'gbdt', 
        objective = 'regression',
        n_jobs = 4,
        is_unbalance = True,
        max_depth = -1,
        max_bin = 512,
        verbose = 0)

    # Create parameters to search
    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [1000],
        'num_leaves': [3, 7, 15],
        'colsample_bytree' : [1]
    }

    # Create the grid
    grid = GridSearchCV(classifier, gridParams, verbose=0, cv=3, n_jobs=1, scoring='neg_mean_absolute_error')
    
    # Run the grid
    print("Training GridSearchCV started...")
    grid.fit(X_train, y_train, eval_metric='mean_absolute_error')
    print("Training GridSearchCV ended...")

    save_grid_results(grid)
    

def save_grid_results(grid):
    # Print the best parameters found
    f = open("reviews_tuning.txt", "w")
    f.write('Best params are: ' + str(grid.best_params_) + '\n')
    f.write('Best score is: ' + str(grid.best_score_) + '\n')
    f.close()


def load_data(filename):
    df = pd.read_csv(filename)
    return df


if __name__ == '__main__':

    # Sample classifier on small data
    filename = 'data/hotels.csv'
    df = load_data(filename)

    # Split train and test set
    X_train, X_test, y_train, y_test = create_feature_sets(df)

    if IS_TUNING:
        train_gridcv(X_train, y_train)
    else:
        # Train classifier
        classifier = train_classifier(X_train, y_train)
        # Evaluate
        evaluate_classifier(classifier, X_test, y_test)
