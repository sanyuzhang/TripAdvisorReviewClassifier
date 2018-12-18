# TripAdvisorReviewClassifier
A LightGBM model based classifier which can predict the overall rating of a hotel by parsing its reviews using NLTK

**Env Setup**
1. Using Python3. All the commands are using python3. If you have both python2 and python3 env, do:
   * Replace python to python3
   * Replace pip to pip3

2. Install all the packages listed in the requirements.txt
   * `pip install -r requirements.txt`

**How to Run**
1. Type the command below.
   * `python reviews_classifier.py fast`
      * fast: Train the processed data with features extracted, which takes about 15 seconds.
   * `python reviews_classifier.py slow`
      * slow: Run the full code including feature extractions using NLTK, which takes 10+ minutes.
