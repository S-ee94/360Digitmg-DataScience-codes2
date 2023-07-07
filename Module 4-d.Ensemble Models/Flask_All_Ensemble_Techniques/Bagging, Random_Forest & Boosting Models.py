'''
CRISP-ML(Q):

Business Understanding: 

Business Problem: Heavy competition in winning Oscar awards
Business Objective: Maximize the Oscar winning chances
Business Constraint: Minimize the Production Cost

Success Criteria:
Business: Increase Oscar winning chances by at least 30%
Machine Learning: Achieve an accuracy of atleast 70%
Economic: Cost benefit of around 300MUSD becasue of excess revenue collection from OTT

Data Understanding: 
Dimensions: 505 rows * 19 cols
Marketing expense	
Production expense	
Multiplex coverage	
Budget	
Movie_length	
Lead_ Actor_Rating	
Lead_Actress_rating	
Director_rating	
Producer_rating	
Critic_rating	
Trailer_views	
3D_available	
Time_taken	
Twitter_hastags	
Genre	
Avg_age_actors	
Num_multiplex	
Collection	
Oscar (Target)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer

from sklearn.model_selection import train_test_split

import joblib
import pickle

from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# pip install xgboost
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate


# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as skmet


# data = pd.read_csv(r"movies_classification.csv")

from sqlalchemy import create_engine

engine = create_engine("mssql://@{server}/{database}?driver={driver}"
                             .format(server = "360DIGITMG\SQLEXPRESS",              # server name
                                   database = "movies_db",                                # database
                                   driver = "ODBC Driver 17 for SQL Server"))            # driver name

# data.to_sql('movies_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


sql = "SELECT * FROM movies_tbl;"

df = pd.read_sql_query(sql, engine)

# df = pd.read_csv(r"movies_classification.csv")

df.head()

df.info()

# AutoEDA
# D-Tale
#########

# pip install dtale
import dtale

d = dtale.show(df)
d.open_browser()



# AutoEDA
# Sweetviz
import sweetviz
my_report = sweetviz.analyze(df)

my_report.show_html('Report1.html')
#########


# Input and Output Split
predictors = df.loc[:, df.columns != "Oscar"]
type(predictors)

predictors


target = df["Oscar"]
type(target)

target

# Segregating Non-Numeric features

categorical_features = predictors.select_dtypes(include = ['object']).columns

categorical_features


# Segregating Numeric features
numeric_features = predictors.select_dtypes(exclude = ['object']).columns

numeric_features


# ## Missing values Analysis
# Checking for Null values
df.isnull().sum()

# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

imputation = preprocessor.fit(predictors)

joblib.dump(imputation, 'meanimpute')


cleandata = pd.DataFrame(imputation.transform(predictors), columns = numeric_features)
cleandata

cleandata.isnull().sum()


# ## Outlier Analysis
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# import seaborn as sns
# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.columns))

clean = winsor.fit(cleandata)

# Save winsorizer model
joblib.dump(clean, 'winsor')


cleandata1 = clean.transform(cleandata)

cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# pip install pyqt5

# ## Scaling with MinMaxScaler

scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata1)

joblib.dump(scale, 'minmax')

scaled_data = pd.DataFrame(scale.transform(cleandata1))
scaled_data


# ## Encoding
# Categorical features

predictors['3D_available'].unique().size
predictors['3D_available'].value_counts()

predictors['Genre'].unique().size
predictors['Genre'].value_counts()


encoding_pipeline = Pipeline([('onehot', OneHotEncoder(drop = 'first'))])

preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

clean =  preprocess_pipeline.fit(predictors)   # Works with categorical features only

joblib.dump(clean, 'encoding')

encode_data = pd.DataFrame(clean.transform(predictors))

encode_data

clean_data = pd.concat([scaled_data, encode_data], axis = 1, ignore_index = True)  # concatenated data will have new sequential index
clean_data


# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(clean_data, target, test_size = 0.2, 
                                                    stratify = target, random_state = 0) 


# # Bagging Classifier Model
# from sklearn.ensemble import BaggingClassifier
# decision tree defined first
# from sklearn import tree
clftree = tree.DecisionTreeClassifier()
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score


bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 500,
                            bootstrap = True, n_jobs = -1, random_state = 42)

# Fit the model 
bagging = bag_clf.fit(X_train, Y_train)

print(confusion_matrix(Y_train, bagging.predict(X_train)))
print(accuracy_score(Y_train, bagging.predict(X_train)))
print('\n')
print(confusion_matrix(Y_test, bagging.predict(X_test)))
print(accuracy_score(Y_test, bagging.predict(X_test)))

# Saving the best model
pickle.dump(bagging, open('baggingmodel.pkl', 'wb'))


# ## Cross Validation implementation
# from sklearn.model_selection import cross_validate

def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X=_X,
                           y=_y,
                           cv=_cv,
                           scoring=_scoring,
                           return_train_score=True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

Bagging_cv_scores = cross_validation(bag_clf, X_train, Y_train, 5)
Bagging_cv_scores

def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        # ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

# get_ipython().run_line_magic('matplotlib', 'inline')
model_name = "Bagging Classifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Bagging_cv_scores["Training Accuracy scores"],
            Bagging_cv_scores["Validation Accuracy scores"])


# ## Random Forest Model
# from sklearn.ensemble import RandomForestClassifier

rf_Model = RandomForestClassifier()


# #### Hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [2, 4]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree
bootstrap = [True, False]


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
n_estimators

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)


# ### Hyperparameter optimization with GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 10, verbose = 1, n_jobs = -1)

rf_Grid.fit(X_train, Y_train)

rf_Grid.best_params_

cv_rf_grid = rf_Grid.best_estimator_


# ## Check Accuracy
# Evaluation on Test Data
test_pred = cv_rf_grid.predict(X_test)

accuracy_test = np.mean(test_pred == Y_test)
accuracy_test

cm = skmet.confusion_matrix(Y_test, test_pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Oscar Nominated', 'Not Nominated'])
cmplot.plot()
cmplot.ax_.set(title = 'Oscar Nomination Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

print (f'Train Accuracy - : {rf_Grid.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test, Y_test):.3f}')


# ### Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose = 2, n_jobs = -1)

rf_Random.fit(X_train, Y_train)

rf_Random.best_params_

cv_rf_random = rf_Random.best_estimator_

# Evaluation on Test Data
test_pred_random = cv_rf_random.predict(X_test)

accuracy_test_random = np.mean(test_pred_random == Y_test)
accuracy_test_random

cm_random = skmet.confusion_matrix(Y_test, test_pred_random)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm_random, display_labels = ['Oscar Nominated', 'Not Nominated'])
cmplot.plot()
cmplot.ax_.set(title = 'Oscar Nomination Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


print (f'Train Accuracy - : {rf_Random.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Random.score(X_test, Y_test):.3f}')


# ## Save the best model from Randomsearch CV approach
pickle.dump(cv_rf_random, open('rfc.pkl', 'wb'))


# ## Cross Validation implementation
# from sklearn.model_selection import cross_validate

def cross_validation(model, _X, _y, _cv = 5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X = _X,
                           y = _y,
                           cv = _cv,
                           scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })


Random_forest_result = cross_validation(cv_rf_random, X_train, Y_train, 5)
Random_forest_result


def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.1, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.1, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


model_name = "RandomForestClassifier"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Random_forest_result["Training Accuracy scores"],
            Random_forest_result["Validation Accuracy scores"])


# #  AdaBoosting
# from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf1 = ada_clf.fit(X_train, Y_train)

predictions = ada_clf1.predict(X_test)

# Evaluation on Testing Data
confusion_matrix(Y_test, predictions)

accuracy_score(Y_test, predictions)

# Evaluation on Training Data
accuracy_score(Y_train, ada_clf1.predict(X_train))

# Saving the best model
pickle.dump(ada_clf1, open('adaboost.pkl','wb'))

## GradientBoosting
# from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()
boost_clf1 = boost_clf.fit(X_train, Y_train)

grad_pred = boost_clf1.predict(X_test)

print(confusion_matrix(Y_test, grad_pred))
print(accuracy_score(Y_test, grad_pred))

print(confusion_matrix(Y_train, boost_clf1.predict(X_train)))
print(accuracy_score(Y_train, boost_clf1.predict(X_train)))


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)

boost_clf_p = boost_clf2.fit(X_train, Y_train)

grad_pred_p = boost_clf_p.predict(X_test)

# Evaluation on Testing Data
print(confusion_matrix(Y_test, grad_pred_p))
print('\n')
print(accuracy_score(Y_test,grad_pred_p))


# Evaluation on Training Data
print(confusion_matrix(Y_train, boost_clf_p.predict(X_train)))
accuracy_score(Y_train, boost_clf_p.predict(X_train))

# Save the ML model
pickle.dump(boost_clf_p, open('gradiantboostparam.pkl', 'wb'))

grad_model_p = pickle.load(open('gradiantboostparam.pkl', 'rb'))


## XGBoosting
# pip install xgboost
# import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth = 5, n_estimators = 10000, 
                            learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)

xgb_clf1 = xgb_clf.fit(X_train, Y_train)

xgb_pred = xgb_clf1.predict(X_test)

# Evaluation on Testing Data
print(confusion_matrix(Y_test, xgb_pred))

accuracy_score(Y_test, xgb_pred)

xgb.plot_importance(xgb_clf)

fi = pd.DataFrame(xgb_clf1.feature_importances_.reshape(1, -1), columns = X_train.columns)
fi

# Save the ML model
pickle.dump(xgb_clf1, open('xgb.pkl', 'wb'))

xgb_model = pickle.load(open('xgb.pkl', 'rb'))


## RandomizedSearchCV for XGB
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

# Grid Search
param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9],}


xgb_RandomGrid = RandomizedSearchCV(estimator = xgb_clf, 
                                    param_distributions = param_test1, 
                                    cv = 5, verbose = 2, n_jobs = -1)

Randomized_search1 = xgb_RandomGrid.fit(X_train, Y_train)

cv_xg_clf = Randomized_search1.best_estimator_
cv_xg_clf

randomized_pred = Randomized_search1.predict(X_test)

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(Y_test, randomized_pred)

Randomized_search1.best_params_

randomized_pred_1 = Randomized_search1.predict(X_train)


# Evaluation on Training Data with model with hyperparameters
accuracy_score(Y_train, randomized_pred_1)

pickle.dump(cv_xg_clf, open('Randomizedsearch_xgb.pkl', 'wb'))
