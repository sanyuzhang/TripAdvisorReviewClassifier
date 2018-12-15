import nltk
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


IS_TUNING = False
LINE_SEPARATOR = '\n'

TARGET = 'overall_ratingsource'
FEATURES = ['city', 'country', 'num_reviews']
NEW_FEATURES = ['neg', 'neu', 'pos', 'compound', 'cleaniness', 'room', 'service', 'location', 'value', 'food', 'cleaniness_var', 'room_var', 'service_var', 'location_var', 'value_var', 'food_var']


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

    features = FEATURES.extend(NEW_FEATURES)
    X = df[features]
    y = df[TARGET]

    return train_test_split(X, y, train_size=0.9, random_state=1)


def gen_review_features(df):
    # Extract features from review

    fileroot = 'data/reviews/'

    # Initialze features
    for feature in NEW_FEATURES:
        df[feature] = 0

    # Iterate review docs
    for doc in df['doc_id']:
        filename = fileroot + doc

        openfile = open(filename,  'r', encoding='utf8', errors='ignore')
        reviews = openfile.read().split(LINE_SEPARATOR)
        openfile.close()

        # Add or fix feature extraction functions below
        neg, neu, pos, compound = sentimental_from_review(reviews)
        df.loc[df['doc_id'] == doc, 'neg'] = neg
        df.loc[df['doc_id'] == doc, 'neu'] = neu
        df.loc[df['doc_id'] == doc, 'pos'] = pos
        df.loc[df['doc_id'] == doc, 'compound'] = compound

        quality = get_quality(reviews)
        df.loc[df['doc_id'] == doc, 'cleaniness'] = quality[0][0]
        df.loc[df['doc_id'] == doc, 'cleaniness_var'] = quality[0][1]
        df.loc[df['doc_id'] == doc, 'room'] = quality[1][0]
        df.loc[df['doc_id'] == doc, 'room_var'] = quality[1][1]
        df.loc[df['doc_id'] == doc, 'service'] = quality[2][0]
        df.loc[df['doc_id'] == doc, 'service_var'] = quality[2][1]
        df.loc[df['doc_id'] == doc, 'location'] = quality[3][0]
        df.loc[df['doc_id'] == doc, 'location_var'] = quality[3][1]
        df.loc[df['doc_id'] == doc, 'value'] = quality[4][0]
        df.loc[df['doc_id'] == doc, 'value_var'] = quality[4][1]
        df.loc[df['doc_id'] == doc, 'food'] = quality[5][0]
        df.loc[df['doc_id'] == doc, 'food_var'] = quality[5][1]

    return df


def sentimental_from_review(reviews):
    neg, neu, pos, compound, num_of_words = 0, 0, 0, 0, 0
    sid = SentimentIntensityAnalyzer()
    for review in reviews:
        tokenized_review = tokenize.sent_tokenize(review)
        num_of_words += len(tokenized_review)
        review_sentiment = sid.polarity_scores(review)
        neg += review_sentiment['neg'] * len(tokenized_review)
        neu += review_sentiment['neu'] * len(tokenized_review)
        pos += review_sentiment['pos'] * len(tokenized_review)
        compound += review_sentiment['compound'] * len(tokenized_review)
    return neg / num_of_words, neu / num_of_words, pos / num_of_words, compound / num_of_words

def get_quality(reviews):
    cleaniness_list, room_list, service_list, location_list, value_list, food_list = [], [], [], [], [], []
    all_num_of_words = 0
    for review in reviews:
        num_of_words = len(tokenize.word_tokenize(review))
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
