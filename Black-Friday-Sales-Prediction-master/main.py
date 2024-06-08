# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_csv("BlackFridaySales.csv")

# Data preprocessing
# Encoding categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Age', 'City_Category']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Convert Stay_In_Current_City_Years to numerical values
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].str.extract('(\d+)')
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(float)

# Filling missing values
data['Product_Category_2'] = data['Product_Category_2'].fillna(data['Product_Category_2'].mean())
data['Product_Category_3'] = data['Product_Category_3'].fillna(data['Product_Category_3'].mean())

# Splitting the data into train and test sets
X = data.drop(columns=['Purchase', 'User_ID', 'Product_ID'])  # Remove non-predictive columns
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling phase
# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBoost Regressor": XGBRegressor()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    results[name] = rmse

# Find the best performing model
best_model = min(results, key=results.get)
best_rmse = results[best_model]

# Print evaluation results
print("Evaluation Results:")
for name, rmse in results.items():
    print(f"{name}: RMSE = {rmse}")

print(f"\nBest Model: {best_model} (RMSE = {best_rmse})")
