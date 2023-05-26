# Sentiment-Analysis
# Sentiment Analysis on Twitter Data

This project focuses on building a system to automatically identify emotional states expressed by people about a company's product on Twitter. The task involves text classification and sentiment analysis.

## Introduction

Text classification is a common Natural Language Processing (NLP) task used for various applications such as customer feedback categorization and support ticket routing. In this project, we aim to classify emotional states (e.g., anger, joy) expressed by people about a company's product on Twitter.

## Installation

To run this project, you need to install the following libraries:
- pandas
- numpy
- re
- string
- matplotlib
- seaborn
- wordcloud
- spacy
- sklearn

## Usage

1. Import the required libraries.
2. Load the training and validation datasets.
3. Perform exploratory data analysis to understand the dataset.
4. Preprocess the data by removing URLs, emojis, and duplicate entries.
5. Train and evaluate various classification models using the preprocessed data.
6. Analyze the results and report performance metrics.
7. Draw conclusions based on the findings.

## Dataset

The dataset consists of Twitter data containing the following columns:
- Tweet_ID: Unique identifier for each tweet
- Entity: The company's name associated with the tweet
- Sentiment: Emotional sentiment expressed in the tweet (e.g., positive, negative, neutral)
- Tweet_content: The text content of the tweet

The training dataset contains 74,682 records, and the validation dataset contains 1,000 records.

## Exploratory Data Analysis

- Displayed the shape of the training and validation datasets.
- Checked for missing values (none found).
- Checked for duplicated values and removed them.
- Performed data cleaning by removing URLs and emojis from the tweet content.
- Calculated the length of each tweet.

## Data Preprocessing

- Removed URLs and emojis from the tweet content.
- Removed duplicate entries from the training dataset.

## Model Training and Evaluation

- Utilized various classification models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Gradient Boosting
  - Extra Trees
  - Ridge Classifier
  - SGD Classifier
  - Support Vector Classifier
  - Multinomial Naive Bayes
- Trained and evaluated each model using the preprocessed data.
- Evaluated the performance of each model using metrics such as accuracy, confusion matrix, and classification report.

# Data Visualization and Data Preprocessing
## Importing Libraries
The necessary libraries for data analysis and machine learning are imported, including pandas, numpy, re, string, matplotlib, seaborn, wordcloud, spacy, and various classifiers from scikit-learn.

## Exploratory Data Analysis (EDA)
EDA is performed on the dataset to gain insights and understand the data's characteristics.

The shape of the train and validation datasets is displayed.
Missing values are checked, and it is found that the "Tweet_content" column has some missing values in the train dataset.
Duplicated values are checked and removed from the train dataset.
URLs and emojis are removed from the tweet content using regular expressions.
The length of tweets is calculated and added as a new feature to both train and validation datasets.
Data visualization is performed using pie charts to show the proportions of different sentiments in the train and validation datasets. Kernel density estimation (KDE) plots are used to visualize the distribution of tweet lengths for different sentiments.
A bar plot is created to show the distribution of tweets per brand and sentiment.
Word clouds are generated separately for positive, negative, neutral, and irrelevant sentiments to visualize the most frequent words.

## Data Preprocessing
Outliers in the tweet length column are removed using the interquartile range (IQR) method.
Text preprocessing is performed using the spaCy library, including tokenization and lemmatization.
Train and validation datasets are split into training and testing subsets using a train-test split function from scikit-learn.
Data representation is done using the TF-IDF vectorization technique to convert text data into numerical features.
The target variable is mapped to numerical values for training and testing.
Next Steps
With the data preprocessed and represented, you can now proceed to the model training and evaluation steps. Several classifiers imported earlier can be used to train models on the transformed data and evaluate their performance using various metrics. Classification reports, confusion matrices, and ROC curves can be generated to analyze the models' effectiveness.

For more details, please refer to the code and comments in the Jupyter Notebook file.

# Machine Learning
## Machine Learning Models
In this step, we tested three different machine learning models for sentiment analysis on Twitter data. The models were evaluated based on their performance metrics, including precision, recall, F1-score, and accuracy.

## MultinomialNB
The Multinomial Naive Bayes model was trained and evaluated using TF-IDF vectorized data. The classification report for this model is as follows:
              precision    recall  f1-score   support

           0       0.70      0.80      0.74      3900
           1       0.65      0.89      0.75      4238
           2       0.84      0.64      0.73      3518
           3       0.94      0.44      0.60      2516

    accuracy                           0.72     14172
   macro avg       0.78      0.69      0.70     14172
weighted avg       0.76      0.72      0.72     14172

## RandomForestClassifier
The Random Forest Classifier model was trained and evaluated using TF-IDF vectorized data. The classification report for this model is as follows:
              precision    recall  f1-score   support

           0       0.85      0.94      0.89      3900
           1       0.92      0.92      0.92      4238
           2       0.93      0.90      0.91      3518
           3       0.97      0.85      0.91      2516

    accuracy                           0.91     14172
   macro avg       0.92      0.90      0.91     14172
weighted avg       0.91      0.91      0.91     14172

## ExtraTreesClassifier
The Extra Trees Classifier model was trained and evaluated using TF-IDF vectorized data. The classification report for this model is as follows:
              precision    recall  f1-score   support

           0       0.86      0.95      0.91      3900
           1       0.94      0.93      0.94      4238
           2       0.95      0.91      0.93      3518
           3       0.98      0.88      0.93      2516

    accuracy                           0.92     14172
   macro avg       0.93      0.92      0.92     14172
weighted avg       0.93      0.92      0.92     14172

## Validation Test
The Extra Trees Classifier model was also tested on a separate validation dataset. The classification report for this test is as follows:

              precision    recall  f1-score   support

           0       0.94      0.97      0.95       277
           1       0.95      0.97      0.96       266
           




















