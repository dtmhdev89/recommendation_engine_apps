from constant import RANDOM_STATE
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from kaggle_hub_dataset import DatasetDownloader

dataset_id = "ashpalsingh1525/imdb-movies-dataset"
path = DatasetDownloader(dataset_id=dataset_id).download()

data = pd.read_csv(os.path.join(path, "imdb_movies.csv"))
print(data.head())
