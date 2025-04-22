from dotenv import load_dotenv
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from flasgger import Swagger
from flask import Flask, Response, request
import json
import argparse
import traceback

app = Flask(__name__)
Swagger(app)


class SearchRagApi:
    """RAG search engine API"""

    load_dotenv()

    def __init__(self, llm_provider="gemini"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        if llm_provider == "openai":
            self.llm_model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                verbose=True,
                max_retries=2
            )
        
        if llm_provider == "gemini":
            self.llm_model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                temperature=0,
                verbose=True,
                max_retries=2
            )

    def load_corpus_datasource(self):
        """Load corpus data source"""

        csv_path = os.path.join(
            "data",
            "contracted_services.csv"
        )
        df_contracted_services = pd.read_csv(csv_path)
        list_documents = df_contracted_services["contracted_services"].to_list()

        return list_documents
    
    def generate_corpus_embeddings(self, list_documents):
        """Convert documents to vector embeddings"""

        documents_embeddings = self.embedding_model.encode(
            list_documents,
            convert_to_numpy=True
        )
        
        return documents_embeddings

    def search_text(self, text_query):
        """Search RAG text"""

        list_documents = self.load_corpus_datasource()
        documents_embeddings = self.generate_corpus_embeddings(list_documents)

        query_embedding = self.generate_query_embeddings(text_query)

        list_retrieved_documents = self.retrieve_documents(
            list_documents=list_documents,
            documents_embeddings=documents_embeddings,
            query_embedding=query_embedding
        )

        answer = self.generate_augmented_response(
            text_query,
            list_retrieved_documents
        )

        return answer
    
    def generate_query_embeddings(self, text_query):
        """Generate embeddings from query"""

        query_embedding = self.embedding_model.encode(
            [text_query],
            convert_to_numpy=True
        )

        return query_embedding
    
    def retrieve_documents(
        self,
        list_documents,
        documents_embeddings,
        query_embedding,
        top_k=2
    ):
        """Search documents in embeddings source"""
        dimension = documents_embeddings.shape[1]
        similar_documents_embeddings = faiss.IndexFlatL2(dimension)
        similar_documents_embeddings.add(documents_embeddings)

        _distance, indexes = similar_documents_embeddings.search(
            query_embedding,
            top_k
        )

        list_result_found = [list_documents[i] for i in indexes[0]]

        return list_result_found

    def generate_augmented_response(
        self,
        text_query,
        list_retrieved_documents
    ):
        """Generate augmented response"""

        chat_prompt_template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """

        prompt = ChatPromptTemplate.from_template(chat_prompt_template)
        rag_chain = (
            {"context": lambda x: list_retrieved_documents, "question": RunnablePassthrough()}
            | prompt
            | self.llm_model
            | StrOutputParser()
        )

        print("rag_chain: ", rag_chain)

        result = rag_chain.invoke(text_query)

        return result

    @staticmethod
    @app.route("/service/search", methods=["POST"])
    def search_text_api():
        """RAG Search API
        ---
        tags:
            - RAG Search
        parameters:
            - name: body
              in: body
              required: true
        responses:
            200:
                description: Success search.
        """

        try:
            json_query = json.loads(request.data)
            searchRagApi = SearchRagApi()
            query = json_query["sentence_query"]
            print("query: ", query)
            json_result = searchRagApi.search_text(query)
            json_result = '{ "result" : "', json_result, '"}'

            return Response(
                json_result,
                mimetype="application/json",
            ), 200
        
        except Exception as e:
            print("Error during search:")
            traceback.print_exc()
            print(e)
            return Response(
                '{"result": "Failure to search: "}',
                mimetype='application/json'
            ), 500


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="System options")

    argparser.add_argument(
        "--use-ngrok",
        action="store_true",
        default=False,
        help="Enable ngrok"
    )

    argparser.add_argument(
        "--no-use-ngrok",  # Optional: for explicitly setting to False
        action="store_false",
        dest="use_ngrok",  # Link to the same 'use_ngrok' argument
        help="Disable ngrok"
    )

    options = argparser.parse_args()

    print("parser: ", options)
    if not options.use_ngrok:
        app.run(debug=True, host="127.0.0.1")
    else:
        print("ngrok enabled")
        from pyngrok import ngrok

        port = 5000
        public_url = ngrok.connect(port)
        print(f"ngrok tunnel available at: {public_url}")

        # Start Flask app
        app.run(port=port)
