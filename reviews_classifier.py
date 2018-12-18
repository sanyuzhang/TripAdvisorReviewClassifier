import nltk
import numpy as np
import pandas as pd
import lightgbm as lgb
import synset_finder as sf
from nltk import tokenize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer


LINE_SEPARATOR = '\n'

# True: Use GridCVSearch to tune the parameters of lightGBM model.
# False: Skip tuning.
IS_TUNING = False

# True: Skip the process of parsing reviews, using parsed data instead.
# False: Need 15 min for NLTK to parse 230,917 reviews.
IS_PROCESSED_DATA = True

TARGET = 'overall_ratingsource'
FEATURES = ['city', 'country', 'num_reviews']
NEW_FEATURES = ['neg', 'neu', 'pos', 'compound', 'cleaniness', 'room', 'service', 'location', 'value', 'food', 'cleaniness_var', 'room_var', 'service_var', 'location_var', 'value_var', 'food_var']

#SERVICE
SERVICE_ASPECT = 'service'
SERVICE_KEYWORDS = {'staff', 'staffs', 'service'}
SERVICE_POS_ADJ = sf.find_all_synsets(['nice', 'excellent', 'good', 'great', 'helpful', 'polite'])
SERVICE_NEG_ADJ = sf.find_all_synsets(['bad', 'unpleasent', 'disordered', 'unhelpful', 'impolite', 'unfriendly'])

#ROOM
ROOM_ASPECT = 'room'
ROOM_KEYWORDS = {'room', 'hotel', 'lobby', 'hall'}
ROOM_POS_ADJ = sf.find_all_synsets(['spacious', 'comfortable'])
ROOM_NEG_ADJ = sf.find_all_synsets(['small', 'uncomfortable'])

#CLEANLINESS
CLEAN_ASPECT = 'cleaniness'
CLEAN_KEYWORDS = {'room', 'hotel', 'lobby', 'hall', 'bed', 'bathroom', 'toilet', 'restroom'}
CLEAN_POS_ADJ = sf.find_synsets('clean')
CLEAN_NEG_ADJ = sf.find_synsets('dirty')

#FOOD
FOOD_ASPECT = 'food'
FOOD_KEYWORDS = {'food', 'lunch', 'dinner', 'breakfast', 'brunch', 'tea', 'cafe'}
FOOD_POS_ADJ = sf.find_synsets('delicious')
FOOD_NEG_ADJ = sf.find_synsets('distasteful')

# LOCATION
LOC_ASPECT = 'location'
LOC_KEYWORDS = {'location', 'view'}
LOC_POS_ADJ = sf.find_all_synsets(['good', 'safe', 'close', 'beautiful'])
LOC_NEG_ADJ = sf.find_all_synsets(['far', 'terrible'])

# VALUE
VALUE_ASPECT = 'value'
VALUE_KEYWORDS = {'price', 'value'}
POS_VALUE = sf.find_synsets('affordable')
NEG_VALUE = sf.find_synsets('expensive')
VALUE_POS_ADJ = sf.find_all_synsets(['good', 'reasonable'])
VALUE_NEG_ADJ = sf.find_synsets('high')


def create_feature_sets(df):
    # Create feature sets

    if IS_PROCESSED_DATA:
        # Load processed data to save time
        df = pd.read_csv('processed_data.csv')
    else:
        # Remove the hotels with num_of_reviews < 0
        df = df[df['num_reviews'] >= 0]

        # Generate features
        df = gen_review_features(df)

        # Encode str type
        for col in FEATURES:
            df[col] = pd.Categorical(df[col])
            df[col] = df[col].cat.codes

        df[df == np.Inf] = np.NaN
        df[df == np.NINF] = np.NaN
        df.fillna(0, inplace=True)

        # Save data to file
        df.to_csv('processed_data.csv')

    ALL_FEATURES = FEATURES + NEW_FEATURES

    X = df[ALL_FEATURES]
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

        df = analyze_reviews(df, doc, reviews)

    return df


def analyze_reviews(df, doc, reviews):
    # Add or fix feature extraction functions below, do not forget to update line 17

    cleaniness_list, room_list, service_list, location_list, value_list, food_list = [], [], [], [], [], []
    neg, neu, pos, compound, num_of_words, all_num_of_words = 0, 0, 0, 0, 0, 0
    sia = SentimentIntensityAnalyzer()

    for review in reviews:
        tokens = tokenize.word_tokenize(review)
        num_of_words = len(tokens)
        all_num_of_words += num_of_words

        review_sentiment = sia.polarity_scores(review)
        neg += review_sentiment['neg'] * num_of_words
        neu += review_sentiment['neu'] * num_of_words
        pos += review_sentiment['pos'] * num_of_words
        compound += review_sentiment['compound'] * num_of_words
        
        bgrams = list(nltk.bigrams(tokens))
        tgrams = list(nltk.trigrams(tokens))

        cleaniness_list.append(aspect_scoring(tokens, bgrams, tgrams, CLEAN_ASPECT) * num_of_words)
        room_list.append(aspect_scoring(tokens, bgrams, tgrams, ROOM_ASPECT) * num_of_words)
        service_list.append(aspect_scoring(tokens, bgrams, tgrams, SERVICE_ASPECT) * num_of_words)
        location_list.append(aspect_scoring(tokens, bgrams, tgrams, LOC_ASPECT) * num_of_words)
        value_list.append(aspect_scoring(tokens, bgrams, tgrams, VALUE_ASPECT) * num_of_words)
        food_list.append(aspect_scoring(tokens, bgrams, tgrams, FOOD_ASPECT) * num_of_words)

    quality = (
        to_quality_pair(cleaniness_list, all_num_of_words), to_quality_pair(room_list, all_num_of_words), 
        to_quality_pair(service_list, all_num_of_words), to_quality_pair(location_list, all_num_of_words), 
        to_quality_pair(value_list, all_num_of_words), to_quality_pair(food_list, all_num_of_words)
    )

    df.loc[df['doc_id'] == doc, 'neg'] = neg / all_num_of_words
    df.loc[df['doc_id'] == doc, 'neu'] = neu / all_num_of_words
    df.loc[df['doc_id'] == doc, 'pos'] = pos / all_num_of_words
    df.loc[df['doc_id'] == doc, 'compound'] = compound / all_num_of_words

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


def to_quality_pair(quality_lsit, normalize_val):
    quality = np.array(quality_lsit) / normalize_val
    return (np.mean(quality), np.var(quality))


def aspect_scoring(tokens, bgrams, tgrams, aspect):
    if aspect == CLEAN_ASPECT:
        for word in tokens:
            if word in CLEAN_POS_ADJ:
                return 1
            elif word in CLEAN_NEG_ADJ:
                return -1
    elif aspect == ROOM_ASPECT:
        for (x, y) in bgrams:
            for word in tokens:
                if word in ROOM_POS_ADJ:
                    return 1
                elif word in ROOM_NEG_ADJ:
                    return -1
    elif aspect == SERVICE_ASPECT:
        for (x, y) in bgrams:
            if y in SERVICE_KEYWORDS and x in SERVICE_POS_ADJ:
                return 1
            elif y in SERVICE_KEYWORDS and x in SERVICE_NEG_ADJ:
                return -1
    elif aspect == LOC_ASPECT:
        for (x, y) in bgrams:
            if y in LOC_KEYWORDS and x in LOC_POS_ADJ:
                return 1
            elif y in LOC_KEYWORDS and x in LOC_NEG_ADJ:
                return -1
        for (x, y, z) in tgrams:
            if x in LOC_KEYWORDS:
                if z in LOC_POS_ADJ:
                    return 1
                elif z in LOC_NEG_ADJ:
                    return -1
    elif aspect == VALUE_ASPECT:
        for word in tokens:
            if word in POS_VALUE:
                return 1
            elif word in NEG_VALUE:
                return -1
        for (x, y) in bgrams:
            if y in VALUE_KEYWORDS and x in VALUE_POS_ADJ:
                return 1
            elif y in VALUE_KEYWORDS and x in VALUE_NEG_ADJ:
                return -1
    elif aspect == FOOD_ASPECT:
        for word in tokens:
            if word in FOOD_NEG_ADJ:
                return -1
            elif word in FOOD_POS_ADJ:
                return 1
    return 0


def train_classifier(X_train, y_train):
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mean_absolute_error',
        'num_leaves': 15,
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
    print('Mean absolute error is:', mean_absolute_error(y_test, y_pred))

    # Write TRUTH-PREDICT comparison results to csv file
    pd.DataFrame({'TRUTH':y_test.values, 'PREDICT':y_pred}).to_csv('prediction.csv', index=False, header=['TRUTH', 'PREDICT'])


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
        'learning_rate': [0.005, 0.001, 0.01],
        'n_estimators': [500, 1000, 2000],
        'num_leaves': [7, 15, 23],
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

