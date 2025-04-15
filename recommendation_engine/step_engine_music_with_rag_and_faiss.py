from kaggle_hub_dataset import DatasetDownloader
import os
import pandas as pd
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import argparse


def args_parser():
    option_parser = argparse.ArgumentParser(
        description="Option to run"
    )

    option_parser.add_argument(
        "--search-mode",
        type=str,
        help="value: album or keyword",
        default="album"
    )

    args = option_parser.parse_args

    return args


def main():
    download_path = DatasetDownloader(
        dataset_id="suraj520/music-dataset-song-information-and-lyrics"
    ).download()

    dataset_path = os.path.join(download_path, "songs.csv")

    data = pd.read_csv(dataset_path)
    print(data.head())

    data["combined_features"] = data.apply(
        lambda row: " ".join([
            row['Artist'],
            row['Album'],
            row['Lyrics']
        ]),
        axis=1
    )

    print("-"*20)
    print(data.head())
    
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        data['combined_features'].tolist(),
        convert_to_numpy=True
    )
    print(embeddings.shape)
    print(embeddings[0])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # base_model = "google/flan-t5-large"
    # alternative: deepseek-ai/DeepSeek-R1 this need higher gpu
    generator = pipeline("text2text-generation", model='google/flan-t5-large')

    def recommend_music(
        albums,
        embeddings,
        faiss_index,
        generator
    ):
        if not albums:
            return "Please select at least one album"
        
        selected_indices = data[data["Album"].isin(albums)].index.tolist()
        
        if not selected_indices:
            return "No matching albums found"

        selected_embeddings = np.mean(
            embeddings[selected_indices],
            axis=0,
            keepdims=True
        )

        _, recommended_indices = faiss_index.search(selected_embeddings, 10)
        recommended_songs = data.iloc[
            recommended_indices[0]
        ][['Name', 'Artist', 'Album']].drop_duplicates()

        song_list = "\n".join(
            (
                f"-{row['Name']} by {row['Artist']} (Album: {row['Album']})"
                for _, row in recommended_songs.iterrows()
            )
        )

        prompt = f"""
        Recommend music based on these songs: \n{song_list}\n\n\
        Give a short explanation
        """
        generated_text = generator(prompt, max_length=100)[0]['generated_text']

        return f"""
        ### Recommended songs:\n{song_list}\n\n
        ### Why these songs?\n{generated_text}
        """

    def gradio_ui(faiss_index, generator, embeddings, unique_albums):
        def btn_recommend_music(albums):
            return recommend_music(
                albums=albums,
                faiss_index=faiss_index,
                generator=generator,
                embeddings=embeddings
            )
        
        with gr.Blocks() as app:
            gr.Markdown("RAG Music Recommendation")
            album_selector = gr.CheckboxGroup(
                label='Select up to 5 albums',
                choices=unique_albums[:100],
                type="value"
            )
            recommend_button = gr.Button("Get recommendation")
            output_text = gr.Markdown()
            recommend_button.click(
                fn=btn_recommend_music,
                inputs=[album_selector],
                outputs=[output_text]
            )
        
        return app
    
    def search_music(query, model, faiss_index):
        if not query:
            return "Please enter a keyword to search"
        
        query_embedding = model.encode([query], convert_to_numpy=True)
        _, recommended_indices = faiss_index.search(query_embedding, 10)
        recommended_songs = data.iloc[
            recommended_indices[0]
        ][['Name', 'Artist', 'Album']].drop_duplicates()

        song_list = "\n".join(
            (
                f"-{row['Name']} by {row['Artist']} (Album: {row['Album']})"
                for _, row in recommended_songs.iterrows()
            )
        )

        prompt = f"""
        Recommend music based on the keyword: {query} \n{song_list}\n\n\
        Give a short explanation
        """
        generated_text = generator(prompt, max_length=100)[0]['generated_text']

        return f"""
        ### Recommended songs:\n{song_list}\n\n
        ### Why these songs?\n{generated_text}
        """

    def gradio_ui_for_music_with_keyword(faiss_index, generator):
        def btn_search_music(query):
            return search_music(query, model, faiss_index)

        with gr.Blocks() as search_app:
            gr.Markdown("Search Based Music Recommendation Engine")
            query_input = gr.Textbox(label="Enter a keyword")
            search_button = gr.Button("Search")
            output_text = gr.Markdown()
            search_button.click(
                fn=btn_search_music,
                inputs=[query_input],
                outputs=[output_text]
            ) 

        return search_app

    unique_albums = data['Album'].dropna().unique().tolist()

    system_option = args_parser()
    gradio_mode = system_option.search_mode

    if gradio_mode == "album":
        gr_app = gradio_ui(
            faiss_index=index,
            generator=generator,
            embeddings=embeddings,
            unique_albums=unique_albums
        )

        gr_app.launch()
    
    if gradio_mode == "keyword":
        keyword_search_app = gradio_ui_for_music_with_keyword(
            faiss_index=index,
            generator=generator
        )

        keyword_search_app.launch()

if __name__ == "__main__":
    main()
