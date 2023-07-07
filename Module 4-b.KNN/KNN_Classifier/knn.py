'''CRISP-ML(Q):
    
Business Problem:
There are a lot of assumptions in the diagnosis pertaining to cancer. In a few cases radiologists, 
pathologists and oncologists go wrong in diagnosing whether tumor is benign (non-cancerous) or malignant (cancerous). 
Hence team of physicians want us to build an AI application which will predict with confidence the presence of cancer 
in a patient. This will serve as a compliment to the physicians.

Business Objective: Maximize Cancer Detection
Business Constraints: Minimize Treatment Cost & Maximize Patient Convenience

Success Criteria: 
Business success criteria: Increase the correct diagnosis of cancer in at least 96% of patients
Machine Learning success criteria: Achieve an accuracy of atleast 98%
Economic success criteria: Reducing medical expenses will improve trust of patients and thereby hospital will see an increase in revenue by atleast 12%

Data Collection:
Data is collected from the hospital for 569 patients. 30 features and 1 label comprise the feature set. 
Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)'''
    
    
# CODE MODULARITY IS EXTREMELY IMPORTANT
# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import sklearn.metrics as skmet
import pickle


# MySQL Database connection

from sqlalchemy import create_engine

cancerdata = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 4-b.KNN\KNN_Classifier\cancerdata.csv")

# Creating engine which connect to MySQL
user = 'root' # user name
pw = 'Seemscrazy1994#' # password
db = 'cancer_db' # database

# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# dumping data into database 
cancerdata.to_sql('cancer', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from cancer'

cancerdf = pd.read_sql_query(sql, con = engine)

print(cancerdf)

# Data Preprocessing & EDA
# converting B to Benign and M to Malignant 
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'B', 'Benign', cancerdf['diagnosis'])
cancerdf['diagnosis'] = np.where(cancerdf['diagnosis'] == 'M', 'Malignant', cancerdf['diagnosis'])

cancerdf.drop(['id'], axis = 1, inplace = True) # Excluding id column
cancerdf.info()   # No missing values observed

cancerdf.describe()

# Seperating input and output variables 
cancerdf_X = pd.DataFrame(cancerdf.iloc[:, 1:])
cancerdf_y = pd.DataFrame(cancerdf.iloc[:, 0])


# EDA and Data Preparation
cancerdf_X.info()

# All numeric features
numeric_features = cancerdf_X.select_dtypes(exclude = ['object']).columns

numeric_features

# Imputation strategy for numeric columns
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean'))])


# All categorical features
categorical_features = cancerdf_X.select_dtypes(include = ['object']).columns

categorical_features


# DataFrameMapper is used to map the given Attribute
# Encoding categorical to numeric variable
categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features,
                                                       OneHotEncoder(drop = 'first'))]))])


# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)])

processed = preprocess_pipeline.fit(cancerdf_X)  # Pass the raw data through pipeline

processed

# Save the defined pipeline
import joblib
joblib.dump(processed, 'processed1')

import os 
os.getcwd()

# Transform the original data using the pipeline defined above
cancerclean = pd.DataFrame(processed.transform(cancerdf_X), columns = cancerdf_X.columns)  # Clean and processed data for Clustering

cancerclean.info()

# new_features = cancerclean.select_dtypes(exclude = ['object']).columns 
# new_features

# Define scaling pipeline
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, cancerclean.columns)]) 

processed2 = preprocess_pipeline2.fit(cancerclean)
processed2

# Save the Scaling pipeline
joblib.dump(processed2, 'processed2')

import os 
os.getcwd()

# Normalized data frame (considering the numerical part of data)

cancerclean_n = pd.DataFrame(processed2.transform(cancerclean), columns = cancerclean.columns)

res = cancerclean_n.describe()
res

# Separating the input and output from the dataset
# X = np.array(cancerclean_n.iloc[:, :]) # Predictors 
Y = np.array(cancerdf_y['diagnosis']) # Target

X_train, X_test, Y_train, Y_test = train_test_split(cancerclean_n, Y,
                                                    test_size = 0.2, random_state = 0)

X_train.shape
X_test.shape

# Model building
knn = KNeighborsClassifier(n_neighbors = 21)

KNN = knn.fit(X_train, Y_train)  # Train the kNN model

# Evaluate the model with train data
pred_train = knn.predict(X_train)  # Predict on train data

pred_train

# Cross table
pd.crosstab(Y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

print(skmet.accuracy_score(Y_train, pred_train))  # Accuracy measure

# Predict the class on test data
pred = knn.predict(X_test)
pred

# Evaluate the model with test data
print(skmet.accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames = ['Predictions']) 

cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])
    
acc
    
# Plotting the data accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")


# Hyperparameter optimization
from sklearn.model_selection import GridSearchCV

k_range = list(range(3, 50, 2))
param_grid = dict(n_neighbors = k_range)
  
# Defining parameter range
grid = GridSearchCV(knn, param_grid, cv = 5, 
                    scoring = 'accuracy', 
                    return_train_score = False, verbose = 1)

help(GridSearchCV)

KNN_new = grid.fit(X_train, Y_train) 

print(KNN_new.best_params_)

accuracy = KNN_new.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

# Predict the class on test data
pred = KNN_new.predict(X_test)
pred

cm = skmet.confusion_matrix(Y_test, pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Benign', 'Malignant'])
cmplot.plot()
cmplot.ax_.set(title = 'Cancer Detection - Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Save the model
knn_best = KNN_new.best_estimator_
pickle.dump(knn_best, open('knn.pkl', 'wb'))

import os
os.getcwd()