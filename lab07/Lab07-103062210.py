
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from IPython.display import display
#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
                 header=None, sep=',', engine='python')
features=['label','cap-shape', 'cap-surface', 'cap-color', 'bruises', 
              'odor', 'gill-attachment', 'gill-spacing', 
              'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 
              'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 
              'stalk-color-below-ring','veil-type','veil-color',
              'ring-number','ring-type','spore-print-color',
              'population','habitat']
df.columns = features
df.describe()


# In[2]:

feature_columns = df.columns[1:]


# In[3]:

from sklearn.preprocessing import LabelEncoder

# encode label first
label_le = LabelEncoder()
df['label'] = label_le.fit_transform(df['label'].values)

# encode categorical features

catego_le = LabelEncoder()

# transform categorical values into numerical values
# be careful that '?' will also be encoded
# we have to replace it to NaN in numerical
num_values = []
for i in features:
    if i is not 'label':
        df[i] = catego_le.fit_transform(df[i].values)
        classes_list = catego_le.classes_.tolist()

        # store the total number of values
        num_values.append(len(classes_list))

        # replace '?' with 'NaN'
        if '?' in classes_list:
            idx = classes_list.index('?')
            df[i] = df[i].replace(idx, np.nan)

display(df.head(15))


# In[4]:

display(df.isnull().sum())


# In[5]:

df_drop_row = df.dropna()
print(df_drop_row.shape)


# In[6]:

print('Original: {}'.format(df.shape))

# drop columns with missing values
df_drop_col = df.dropna(axis=1)
print('Drop column: {}'.format(df_drop_col.shape))

# only drop rows where all columns are NaN
df_drop_row_all = df.dropna(how='all')
print('Drop row all: {}'.format(df_drop_row_all.shape))

# drop rows that have not at least 14 non-NaN values
df_drop_row_thresh = df.dropna(thresh=14)
print('Drop row 14: {}'.format(df_drop_row_thresh.shape))


# In[7]:

catego_features_idx = []
for str in feature_columns:
    catego_features_idx.append(df.columns.tolist().index(str)-1)


# In[8]:

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
df_small = df.sample(n=2000, random_state=0)

X = df_small.drop('label', 1)
y = df_small['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# define pipeline with an arbitrary number of transformer in a tuple array
pipe_knn = Pipeline([('imr', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                     ('ohe', OneHotEncoder(categorical_features=catego_features_idx, 
                                           n_values=num_values, sparse=False)),
                     ('scl', StandardScaler()),
                     ('clf', KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski'))])

pipe_svm = Pipeline([('imr', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                     ('ohe', OneHotEncoder(categorical_features=catego_features_idx, 
                                           n_values=num_values, sparse=False)),
                     ('scl', StandardScaler()),
                     ('clf', SVC(kernel='rbf', random_state=0, gamma=0.001, C=100.0))])

# use the pipeline model to train
pipe_knn.fit(X_train, y_train)
y_pred = pipe_knn.predict(X_test)
print('[KNN]')
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('\n[SVC]')
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))


# In[9]:

idx = np.isnan(X_train).sum(1) == 0
X_train = X_train[idx]
y_train = y_train[idx]
idx = np.isnan(X_test).sum(1) == 0
X_test = X_test[idx]
y_test = y_test[idx]
pipe_knn = Pipeline([('ohe', OneHotEncoder(categorical_features = catego_features_idx, 
                                           n_values = num_values, sparse=False)),
                     ('scl', StandardScaler()),
                     ('clf', KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski'))])

pipe_svm = Pipeline([('ohe', OneHotEncoder(categorical_features = catego_features_idx, 
                                           n_values = num_values, sparse=False)),
                     ('scl', StandardScaler()),
                     ('clf', SVC(kernel='rbf', random_state=0, gamma=0.001, C=100.0))])

# use the pipeline model to train
pipe_knn.fit(X_train, y_train)
y_pred = pipe_knn.predict(X_test)
print('[KNN: drop row]')
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))

pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('\n[SVC: drop row]')
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))


# In[10]:

from sklearn.model_selection import GridSearchCV
pipe_svm = Pipeline([('ohe', OneHotEncoder(categorical_features = catego_features_idx, 
                                           n_values = num_values, sparse=False)),
                     ('scl', StandardScaler()),
                     ('clf', SVC(random_state=0))])

param_gamma = [0.0001, 0.001, 0.01, 0.1, 1.0]
param_C = [0.1, 1.0, 10.0, 100.0]

# here you can set parameter for different steps 
# by adding two underlines (__) between step name and parameter name
param_grid = [{'clf__C': param_C, 
               'clf__kernel': ['linear']},
              {'clf__C': param_C, 
               'clf__gamma': param_gamma, 
               'clf__kernel': ['rbf']}]

# set pipe_svm as the estimator
gs = GridSearchCV(estimator=pipe_svm, 
                  param_grid=param_grid, 
                  scoring='accuracy')

gs = gs.fit(X_train, y_train)
print('[SVC: grid search]')
print('Validation accuracy: %.3f' % gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))


# In[11]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=200, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy per feature: %.2f' % (accuracy_score(y_test, y_pred)/len(X.columns)))

importances = forest.feature_importances_
# get sort indices in descending order
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            X.columns.values[indices[f]-1], 
                            importances[indices[f]-1]))


# In[14]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=200, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[ ]:



