import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("--- Step 1: Importing the dataset ---")
# Loading California Housing dataset into a Pandas DataFrame
data = fetch_california_housing(as_frame=True)
df = data.frame
print(f"Dataset successfully loaded. Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\n--- Step 2: Splitting data into train and test sets ---")
# X contains all feature columns, y contains the target price column
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Split data: 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

print("\n--- Step 3: Fitting Linear Regression model & Evaluating ---")
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Model R-squared (R²) Score: {r2:.4f}")
print(f"Mean Squared Error (MSE):   {mse:.4f}")

print("\n--- Step 4: Saving the trained model ---")
# Save model to disk using joblib
model_filename = 'house_price_model.pkl'
joblib.dump(model, model_filename)
print(f"Success! Saved trained model as '{model_filename}'")