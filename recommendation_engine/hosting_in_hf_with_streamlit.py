from step_engine_movie_recommendation import DataPreprocess, RecommendationModel, \
    get_recommendations, get_movies_by_genre
import streamlit as st


if __name__ == "__main__":
    dataset_url = "https://huggingface.co/spaces/hieudev/recommendationsystem/resolve/main/imdb_movies.csv"
    try:
        data = DataPreprocess().perform(dataset_url=dataset_url)
        st.write("Dataset Loaded Successfully")
        st.write(data.head())
    except Exception as e:
        st.write(f"Failed to load dataset: {e}")
        st.stop()

    model = RecommendationModel(data=data).model

    st.title("Movie Recommendation System")
    genre_list = sorted(data['genre'].unique().tolist())
    selected_genre = st.selectbox("Select a Genre", genre_list)
    movie_options = get_movies_by_genre(data, selected_genre)
    selected_movies = st.multiselect("Select Up to 3 Movies", movie_options)

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(
            model,
            data,
            selected_movies,
            selected_genre
        )
        st.write("Recommended Movies")
        for movie in recommendations:
            st.write(f"- {movie}")
