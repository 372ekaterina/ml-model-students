import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
data_train = pd.read_csv('data/Student_performance_data _.csv')
df_train = pd.DataFrame(data_train)


# MODEL
# Separate the features 
y = data_train['GradeClass']
feature_columns = ['Ethnicity', 'Gender', 'Age', 'ParentalSupport']
X = data_train[feature_columns]

# Split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)

# Specify the model
students_model = DecisionTreeRegressor(random_state=1)

# Fit model with the training data.
students_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = students_model.predict(val_X)

# Calculate the Mean Absolute Error in Validation Data
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

# Save CSV file
val_predictions.to_csv('data/best_model_submission.csv', index=False)
