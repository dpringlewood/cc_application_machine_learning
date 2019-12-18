import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings(action='ignore')


# Read in the data to a pandas DataFrame
cc_apps = pd.read_csv('./cc_approvals.data', header=None)

# Clean the data, it has ? in place of null values.
cc_apps.replace('?', np.nan, inplace=True)
cc_apps.fillna(cc_apps.mean(), inplace=True)
for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Preprocess the data using sklearn
# The LabelEncoder takes non-numeric values and converts them to
# numeric value so that sklearn can crunch over it
le = LabelEncoder()

for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps[col]=le.fit_transform(cc_apps[col].values)

cc_apps.drop([11, 13], axis=1, inplace=True)
cc_apps = cc_apps.to_numpy()

# Split the data up into training and test sets
X ,y = cc_apps[:, 0:13], cc_apps[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)
# Need to scale some of the numeric data
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X_train = scaler.fit_transform(X_train)
rescaled_X_test = scaler.fit_transform(X_test)

# Using a logistic regression model to classify cc_applications
logreg = LogisticRegression()
logreg.fit(rescaled_X_train, y_train)

y_pred = logreg.predict(rescaled_X_test)

print('The accuracy of the classifier is: ', logreg.score(rescaled_X_test, y_test))
print(confusion_matrix(y_test, y_pred))

tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)

# Using GridSearchCV enhances the accuracy of the model
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=2)
rescaled_X = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaled_X, y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print(('Best: %f using%s' %(best_score, best_params)))

