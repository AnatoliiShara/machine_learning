import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

# create a synthetic dataset with 10 features
np.random.seed(42)

data = {
	'categorical_feature': np.random.choice(["A", "B", "C"], size=1000),
	'boolean_feature': np.random.choice([True, False], size=1000),
	'integer_feature': np.random.randint(1, 100, size=1000),
	'float_feature': np.random.rand(1000),
	'feature_5': np.random.choice(["X", "Y", "Z"], size=1000),
	'feature_6': np.random.choice([0,1,2], size=1000),
	'feature_7': np.random.uniform(0,10, size=1000), 
	'feature_8': np.random.choice([10, 20, 30], size=1000),
	'feature_9': np.random.normal(0,1, size=1000),
	'feature_10': np.random.choice([0.1, 0.2, 0.3], size=1000)
}

df = pd.DataFrame(data)
#print(df.head(5))

# create a target variable
df['target'] = 3 * df['integer_feature'] + 5 * df['float_feature'] + np.random.normal(0,1, size=1000)

# preprocessing
categorical_features = ['categorical_feature', 'boolean_feature', 'feature_5']
numeric_columns = ['integer_feature', 'float_feature', 'feature_6', 
						'feature_7', 'feature_8', 'feature_9', 'feature_10']

# Label Encoder for categorical features
le = LabelEncoder()
for col in categorical_features:
	df[col] = le.fit_transform(df[col])

# train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standard scaling for numeric features
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Linear Regression

model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"mean_squared_error: {mse}")

# simple testing

# Testing
sample_data = {
    'categorical_feature': ['A'],
    'boolean_feature': [True],
    'integer_feature': [50],
    'float_feature': [0.75],
    'feature_5': ['X'],
    'feature_6': [1],
    'feature_7': [5.0],
    'feature_8': [20],
    'feature_9': [0.2],
    'feature_10': [0.2]
}

sample_df = pd.DataFrame(sample_data)
sample_df[categorical_features] = le.transform(sample_df[categorical_features])
sample_df[numeric_columns] = scaler.transform(sample_df[numeric_columns])

sample_prediction = model.predict(sample_df)
print(f"Sample Prediction: {sample_prediction[0]}")































































