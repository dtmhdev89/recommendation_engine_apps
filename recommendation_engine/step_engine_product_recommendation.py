import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from step_feature_selection import raw_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


new_data = raw_data.drop(columns=["Size"]).copy()
new_data = new_data[new_data["Rating"] >= 4.0].reset_index()

user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
encoders = {
    "user": user_encoder,
    "product": product_encoder
}
new_data["User ID"] = user_encoder.fit_transform(new_data["User ID"])
new_data["Product ID"] = product_encoder.fit_transform(new_data["Product ID"])

num_users = new_data["User ID"].nunique()
num_products = new_data["Product ID"].nunique()

user_input = Input(shape=(1,))
product_input = Input(shape=(1,))

user_embedding = Embedding(num_users + 1, 50)(user_input)
product_embedding = Embedding(num_products + 1, 50)(product_input)
dot_product = Dot(axes=1)(
    [
        Flatten()(user_embedding),
        Flatten()(product_embedding)
    ]
)

output = Dense(1, activation="linear")(dot_product)

model = Model(
    inputs=[user_input, product_input],
    outputs=output
)

model.compile(loss="mse", optimizer="adam")
user_product_pairs = new_data[["User ID", "Product ID"]].values
ratings = new_data["Rating"].values

model.fit(
    [user_product_pairs[:, 0], user_product_pairs[:, 1]],
    ratings,
    epochs=10,
    batch_size=32,
    verbose=1
)


def recommend_product(
    encoders,
    user_id,
    data,
    model,
    top_n=5
):
    encoded_user_id = encoders["user"].transform([user_id])
    product_ids = data["Product ID"].unique()
    prediction = model.predict(
        [np.full_like(product_ids, encoded_user_id), product_ids]
    )
    top_indices = prediction.flatten().argsort()[-top_n:][::-1]
    recommended_products = data.iloc[top_indices][["Product Name", "Brand", "Category", "Price", "Color", "Rating"]]

    return recommended_products


user_id_example = 97
recommendations = recommend_product(
    encoders=encoders,
    user_id=user_id_example,
    data=new_data,
    model=model
)

print(recommendations)

# with TF-IDF
new_data["combined_features"] = new_data["Product Name"] + " " + \
    new_data["Brand"] + " " + new_data["Category"] + \
    new_data["Color"]

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(new_data["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_product_tfidf(
    user_id,
    data,
    cosine_sim,
    top_n=5
):
    purchased_products = data[data["User ID"] == user_id]["Product ID"].values
    if len(purchased_products) == 0:
        return [["No purchase history found", ""]]
    
    last_purchased_product_id = purchased_products[-1]
    product_idx = data[data["Product ID"] == last_purchased_product_id].index[0]
    similar_indices = cosine_sim[product_idx].argsort()[-(top_n + 1): -1][::-1]
    recommended_products = data.iloc[similar_indices][["Product Name", "Brand"]]

    return recommended_products.values.tolist()


user_id_example = 25
recommendation_tfidf = recommend_product_tfidf(
    user_id=user_id_example,
    data=new_data,
    cosine_sim=cosine_sim
)

print(recommendation_tfidf)
