from constant import RANDOM_STATE
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import gradio as gr
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from collections import defaultdict

from kaggle_hub_dataset import DatasetDownloader

dataset_id = "ashpalsingh1525/imdb-movies-dataset"
path = DatasetDownloader(dataset_id=dataset_id).download()

original_data = pd.read_csv(os.path.join(path, "imdb_movies.csv"))
data = original_data.copy()

features = [
    "genre", "crew", "orig_title", "status", "orig_lang",
    "budget_x", "revenue", "country"
]
encoding_features = [
    "genre", "crew", "orig_title",
    "status", "orig_lang", "country"
]
target = "score"
label_encoders = dict()

for col in encoding_features:
    le = LabelEncoder()
    original_data[col] = le.fit_transform(original_data[col].astype(str))
    label_encoders[col] = le

imputer = SimpleImputer(strategy="median")
original_data[features] = imputer.fit_transform(original_data[features])

X = original_data[features]
y = original_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df)

categorical_columns = ['genre', 'orig_title', 'orig_lang', 'country', 'crew']

for col in categorical_columns:
    data[col] = data[col].astype(str)

if data['genre'].str.isnumeric().all():
    print("----Warning: Genre column is numeric. Mapping needed")
    genre_mapping = {i: f"Genre_{i}" for i in data["genre"].unique()}
    data['genre'] = data["genre"].map(genre_mapping)

print("*"*20)
print("--Data after mapping categorical columns")
print(data.head())

reader = Reader(rating_scale=(data["score"].min(), data["score"].max()))
dataset = Dataset.load_from_df(
    data[['orig_title', 'orig_lang', 'score']],
    reader
)

trainset, testset = surprise_train_test_split(
                        dataset,
                        test_size=0.2,
                        random_state=RANDOM_STATE
                    )

# print("--Data after using Dataset to load")
# for uid, iid, rating in trainset.all_ratings():
#     print(f"User ID (inner): {uid}, Item ID (inner): {iid}, Rating: {rating}")
#     # You can stop after a few iterations for demonstration
#     if uid > 5:
#         break

# Singular Value Decomposition (SVD) a popular matrix factorization technique
# used for collaborative filtering in recommender systems
# n_factors=50 => each user and item is represented in a 50-dimensional latent space
model = SVD(n_factors=50, random_state=RANDOM_STATE)
model.fit(trainset)


def build_gradio_interface(model, data):
    def get_recommendations(selected_movies, genre):
        if not selected_movies:
            return "Please select at least one movie"
        
        filter_movies = data[data['genre'] == genre]
        movie_scores = defaultdict(float)
        for movie in filter_movies['orig_title'].unique():
            est_score = model.predict(uid="user", iid=movie).est
            movie_scores[movie] = est_score

        recommended_movies = sorted(
            movie_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommended_movies = [movie for movie, _ in recommended_movies if movie not in selected_movies]

        return recommended_movies[:5]

    def get_movie_by_genre(genre):
        return data[data['genre'] == genre]['orig_title'].unique().tolist()

    def movie_recommeder_ui(genre, selected_movies):
        recommendations = get_recommendations(selected_movies, genre)
        
        return recommendations
    
    with gr.Blocks() as demo:
        gr.Markdown("Move Recommended System")
        genre_dropdown = gr.Dropdown(
            choices=sorted(data['genre'].unique().tolist()),
            label="Select a genre"
        )
        movie_list = gr.CheckboxGroup(choices=[], label="Select up to 3 Movies")
        recommend_button = gr.Button("Get Recommendation")
        output_text = gr.Textbox(label="Recommended Movies")

        def update_movies(genre):
            return gr.update(choices=get_movie_by_genre(genre), value=[])
        
        genre_dropdown.change(
            update_movies,
            inputs=[genre_dropdown],
            outputs=[movie_list]
        )

        recommend_button.click(
            movie_recommeder_ui,
            inputs=[genre_dropdown, movie_list],
            outputs=[output_text]
        )

    return demo


gr_demo = build_gradio_interface(model, data)
gr_demo.launch()
