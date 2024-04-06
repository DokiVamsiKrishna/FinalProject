# Import Libraries and modules

# libraries that are used for analysis and visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
# Impoting data preprocessing libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importing model selection libraries.
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Importing metrics for model evaluation.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Importing machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# Importing SMOTE for handling class imbalance.
from imblearn.over_sampling import SMOTE

# Importing warnings library. Would help to throw away warnings caused.
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("cardiovascular_risk.csv")
# Missing Values Percentage
round(df.isna().sum()/len(df)*100, 2)
# Missing Values Percentage
round(df.isna().sum()/len(df)*100, 2)
# features which has less than 5%  null values present. 
nan_columns = ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate']

# dropping null values
df.dropna(subset=nan_columns, inplace=True)
# glucose level are continuous in nature. 
# Outlier are not treating yet thats why imputimg NaN values with median value.

df['glucose'] = df.glucose.fillna(df.glucose.median())
     

# checking for null values after treating them.
df.isna().sum()
# Label Encoding

df['sex'] = df['sex'].map({'M':1, 'F':0})
df['is_smoking'] = df['is_smoking'].map({'YES':1, 'NO':0})
# one-hot encode the 'education' feature
education_onehot = pd.get_dummies(df['education'], prefix='education')

# drop the original education feature
df.drop('education', axis=1, inplace=True)

# concatenate the one-hot encoded education feature with the rest of the data
df = pd.concat([df, education_onehot], axis=1)
# adding new column PulsePressure 
df['pulse_pressure'] = df['sysBP'] - df['diaBP']

# dropping the sysBP and diaBP columns
df.drop(columns=['sysBP', 'diaBP'], inplace=True)
     
#If a person smokes (is_smoking=='yes'), but the number of cigarettes smoked per day is 0, or cigsPerDay is 0. Then it may develop into a conflicting case, we must treat those records.

# checking data, weather the provide information is correct or not
df[(df.is_smoking == 'YES') & (df.cigsPerDay == 0)]
# droping is_smoking column due to multi-collinearity

df.drop('is_smoking', axis=1, inplace=True)
X = df.drop('TenYearCHD', axis=1)
y= df['TenYearCHD']
# importing libarary
from sklearn.ensemble import ExtraTreesClassifier

# model fitting 
model = ExtraTreesClassifier()
model.fit(X,y)

# ranking feature based on importance
ranked_features = pd.Series(model.feature_importances_,index=X.columns)
# importing libarary
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# model fitting
ordered_rank_features = SelectKBest(score_func=chi2, k='all')
model = ordered_rank_features.fit(X,y)

# ranking feature based on importance
feature_imp = pd.Series(model.scores_,index=X.columns)
feature_imp.sort_values(ascending=False)
# importing libarary
from sklearn.feature_selection import mutual_info_classif

# model fitting
mutual_info = mutual_info_classif(X,y)

# ranking feature based on importance. 
mutual_data = pd.Series(mutual_info, index=X.columns)
mutual_data.sort_values(ascending=False)
# copying the data to save the work done till now
model_df = df.copy()
model_df=model_df.drop(columns=['id','education_1.0','education_2.0','education_3.0','education_4.0','prevalentHyp','BPMeds'])
dependent_variable = 'TenYearCHD'
independent_variable = list(model_df.columns)
independent_variable.remove(dependent_variable)

X =df[independent_variable].values
y =df[dependent_variable].values

## Handling target class imbalance using SMOTE
from collections import Counter
from imblearn.over_sampling import SMOTE

# Resampling the minority class
smote = SMOTE(random_state=42)
# fit predictor and target variable
X, y = smote.fit_resample(X, y)
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import StackingClassifier
# Create individual classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
estimators = [
    ('logistic', LogisticRegression(random_state=42)),
    ('tree', DecisionTreeClassifier(random_state=42)),
    ('rf', KNeighborsClassifier(n_neighbors=3)),
    ('sv',SVC(random_state=42))
]

# Create the StackingClassifier
stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=SVC()  # You can choose any final estimator
)

# Fit the StackingClassifier to the training data
stacking_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = stacking_classifier.predict(X_test)

pickle.dump(stacking_classifier, open('model.pkl', 'wb'))
pickled_model=pickle.load(open('model.pkl','rb'))
pickled_model.predict(X_test)

