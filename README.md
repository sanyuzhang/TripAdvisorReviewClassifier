# TripAdvisorReviewRegressor
A LightGBM model based regressor which can predict the overall rating of a hotel by parsing its reviews using NLTK

**Env Setup**
1. Using Python3. All the commands are using python3.
2. If you have both python2 and python3 env, do:
   * Replace python to python3
   * Replace pip to pip3

2. Install all the packages listed in the requirements.txt
   * `pip install -r requirements.txt`

**How to Run**
1. Type the command below.
   * `python reviews_classifier.py`
      * Train the processed data with features extracted, which takes about 10+ seconds.
   * `python reviews_classifier.py full`
      * Run the full code including feature extractions using NLTK, which takes 10+ minutes.

**Folders**
1. All the raw data are under the `data/`
2. `processed_data.csv` are data of features extracted from raw data using NLTK.
3. `prediction.csv` are a table of true ratings and our predicted ratings.
