<h1 align="center">Disaster Tweet Classification App</h1>

# Overview: 

The Disaster Tweet Claasification Application classifies whether a tweet indicates an occured disaster or not.  
This Application processess the text removing URLs, HTML Tags, Emojis and other Non-ASCII characters, punctuations and stopwords. Furthermore, the tweets are tokenized, Lemmatized and Encode them as part of Feature Enigneering.  
A custom desinged Classifier built with Multinomial Naive Bayes is used for prediction.

# Project Planning:

## 1.  Data Exploration
  * Distribution of target variable.
  * Visualization of various characteristics in Disaster and Non-disaster Tweets.
  * Looking at Tweets analyzing the necessary cleaning strategies.  

<img
  src="https://github.com/Praveen-Samudrala/End-2-End-Disaster-Tweet-Classification-App-NLP/blob/main/images/data1.png"
  alt="Data_Sample"
  title="Data Sample"/>  

## 2. Data Preprocessing  
  * Cleans the input text by removing URLs, special characters, emojis, HTML tags,
    punctuation, and numbers.
  * Expands common chat abbreviations into their full forms based on a predefined dictionary.
  * Converts NLTK part-of-speech tags to WordNet format.
  * Removing stopwords
  * Lemmatization.
  * Feature Encoding - Experimented with BagofWords and TF-IDF Vectorization.

## 3. Model Building and Evaluation
  * Model Experimentation 1
    - Support Vector Machines
    - Logistic Regression
    - Multinomial Naive Bayes
    - Decision Trees
    - Random Forest Classifier
    - LightGBM
    - CatBoost
    - XGBoost

  * Model Experimentation 2
    - Multinomial Naive Bayes with Count Vectorizer

  * Model Experimentation 3
    - Hyperparameter Optimization

  * Evaluation

<img
  src="https://github.com/Praveen-Samudrala/End-2-End-Disaster-Tweet-Classification-App-NLP/blob/main/images/performance.png"
  alt="performance"
  title="performance"/>

<img
  src="https://github.com/Praveen-Samudrala/End-2-End-Disaster-Tweet-Classification-App-NLP/blob/main/images/performance1.png"
  alt="performance2"
  title="performance2"/>

## 4. Inference function
  * Training a model on whole dataset (X and y) without splitting.
  * Function "assess_customer" to assess new customers whether they payback loan or not.
  * Check model predictions on new customer.

## 5. Prediction App with Streamlit
<img
  src="https://github.com/omrfrkaytnc/disaster_predictor_app_nlp/blob/main/images/homepage.jpeg"
  alt="Home Page_1"
  title="Home Page"/>

<img
  src="(https://github.com/omrfrkaytnc/disaster_predictor_app_nlp/blob/main/images/homepage2.png)"
  alt="Home Page_2"
  title="Home Page"/>

<img
  src="(https://github.com/omrfrkaytnc/disaster_predictor_app_nlp/blob/main/images/homepage3.png)"
  alt="Home Page_3"
  title="Home Page"/>
<img
  src="(https://github.com/omrfrkaytnc/disaster_predictor_app_nlp/blob/main/images/classifier_disaster.jpeg)"
  alt="Result_1"
  title="Home Page"/>

<img
  src="( https://github.com/omrfrkaytnc/disaster_predictor_app_nlp/blob/main/images/classifier_not_disaster.jpeg)"
  alt="Result_0"
  title="Home Page"/>
