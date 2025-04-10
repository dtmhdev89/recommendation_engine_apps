from constant import RANDOM_STATE
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from kaggle_hub_dataset import DatasetDownloader

path = DatasetDownloader().download()
raw_data = pd.read_csv(os.path.join(path, "fashion_products.csv"))
data = raw_data.copy()

features = ["Brand", "Category", "Price", "Color", "Size"]
target = "Rating"

label_encoders = dict()

for col in list(filter(lambda x: x != "Price", features)):
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Handling missing values
imputer = SimpleImputer(strategy="median")
data[features] = imputer.fit_transform(data[features])

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
feature_importances = pd.DataFrame(
    {
        "Feature": features,
        "Importance": model.feature_importances_
    }
)

feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("---Feature Importance")
print(feature_importances)
