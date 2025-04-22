import argparse
import json
from domain.purchase.PurchaseOrderPreprocessingDomain import PurchaseOrderPreprocessingDomain
from domain.purchase.PurchaseOrderWordEmbeddingsDomain import PurchaseOrderWordEmbeddingsDomain
from domain.purchase.PurchaseOrderDomain import PurchaseOrderDomain
from flask import Flask, Response, request
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


class SemanticSearchApi:
    """Semantic Search API"""

    @staticmethod
    @app.route("/")
    def get():
        return "Ok"
    
    @staticmethod
    @app.route("/purchaseorder/preprocessing", methods=["POST"])
    def pre_processing_purchase_order():
        """
        Endpoint for data preprocessing before ML model execution.
        ---
        tags:
          - Data preprocessing
        responses:
            201:
                description: Success preprocessing file.
            500:
                description: Failure preprocessing file.
        """

        try:
            purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
            purchaseOrderPreprocessingDomain.data_preprocessing()

            return Response(
                '{"result": "Success to preprocessed the Deataset."}',
                mimetype='application/json'
            ), 201

        except Exception as e:
            print(e)
            return Response(
                {"result": "Failure to preprocessed the Deataset."},
                mimetype='application/json'
            ), 500

    @staticmethod
    @app.route("/purchaseorder/embeddings", methods=["POST"])
    def create_word_embeddings_file():
        """Generate Word Embeddings - Endpoint to generate word embeddings before ML execution
        ---
        tags:
            - Word Embeddings
        responses:
            201:
                description: Success to generate embeddings file
            500:
                description: Failure to generate embeddings file
        """

        try:
            purchaseOrderWordEmbeddingsDoamin = PurchaseOrderWordEmbeddingsDomain()
            purchaseOrderWordEmbeddingsDoamin.create_word_embeddings_file()

            return Response(
                '{"result": "Success to generate Text Embeddings file."}',
                mimetype='application/json'
            ), 201
        except Exception as e:
            print(e)
            return Response(
                {"result": "Failure to generate the Text Embeddings file"}
            ), 500

    @staticmethod
    @app.route("/purchaseorder/semanticsearch", methods=["POST"])
    def semantic_search():
        """
        Semantic search - Endpoint to semantic search
        ---
        tags:
            - Semantic Search
        responses:
            201:
                description: Success to find records
        parameters:
            - name: body
              in: body
              required: true
        """

        try:
            json_query = json.load(request.data)
            purchaseOrderDomain = PurchaseOrderDomain()
            json_result = purchaseOrderDomain.semantic_search(json_query["sentence_query"])
            json_result = json_result.to_json(orient="records")

            return Response(
                json_result,
                mimetype="application/json"
            ), 200
        except Exception as e:
            print(e)
            return Response(
                '{"result": "Failure to search"}',
                mimetype="application/json"
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
