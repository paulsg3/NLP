# Importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import os
from sklearn.metrics import confusion_matrix, f1_score

# Defining file paths
# Change this path (the one directly below) to wherever you have saved this file and the reviews.csv file

os.chdir('/Users/paulshingay/Desktop/Desktop/Schulich/MMAI_5400/Assignment_2/')

# Defining the file paths to where the train and validation data will be saved

cwd = os.getcwd()

train_path = cwd + "/train.csv"
validation_path = cwd + "/valid.csv"

data = pd.read_csv(
    'reviews.csv', sep='\t')

data.loc[data['RatingValue'] == 1, 'Sentiment'] = 0
data.loc[data['RatingValue'] == 2, 'Sentiment'] = 0
data.loc[data['RatingValue'] == 3, 'Sentiment'] = 1
data.loc[data['RatingValue'] == 4, 'Sentiment'] = 2
data.loc[data['RatingValue'] == 5, 'Sentiment'] = 2

data.sort_values(by='Sentiment', ascending=False, inplace=True)
data.reset_index(inplace=True)
del data['index']

# Creating balanced dataset

data_df = data.tail(683)

data_df.reset_index(inplace=True)
del data_df['index']

# Splitting the data into training and validation sets

X_data = data_df['Review']
y_data = data_df['Sentiment']

X_train, X_validation, y_train, y_validation = train_test_split(
    X_data, y_data, test_size=0.2, random_state=12345)

train_df = pd.concat([X_train, y_train], axis=1)
validation_df = pd.concat([X_validation, y_validation], axis=1)

# Saving train.csv and valid.csv (paths defined above)

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)

# Loading train.csv and valid.csv

train_data = pd.read_csv(train_path)

validation_data = pd.read_csv(validation_path)

X_train = train_data['Review']
y_train = train_data['Sentiment']

X_val = validation_data['Review']
y_val = validation_data['Sentiment']


# SGD with Grid Search (model with the best performance)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])


parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(X_train, y_train)

gs_predicted = gs_clf.predict(X_val)

print('\n')
print('Accuracy: ', gs_clf.best_score_)
print('\n')
print('Average F1 score: ', f1_score(y_val, gs_predicted, average='macro'))
print('\n')

# Confusion matrix

cm = confusion_matrix(y_val, gs_predicted)
cm_normalized = cm/cm.sum(axis=1)
label_names = ['negative', 'neutral', 'positive']
cm_df = pd.DataFrame(cm_normalized, index=label_names, columns=label_names)

target_names = ['negative', 'neutral', 'positive']
print('Class-Wise F1-Scores:')
print('\n')
print(metrics.classification_report(
    y_val, gs_predicted, target_names=target_names))

print('\n')
print('Normalized Confusion Matrix:')
print('\n')
print(cm_df)
print('\n')
