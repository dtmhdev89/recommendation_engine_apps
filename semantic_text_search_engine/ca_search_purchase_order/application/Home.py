from domain.purchase.PurchaseOrderRepository import PurchaseOrderRepository
import json
from flask import Flask, Response, request
from flasgger import Swagger


app = Flask(__name__)
print(f"The name of the Flask application is: {app.name}")
Swagger(app)


class SemanticSearchTestApi():

    @staticmethod
    @app.route("/")
    def welcome():
        """
        Home page
        ---
        tags:
           - Welcome page
        responses:
            200:
                description: Success to home.
            500:
                description: Failure to go to home.
        """
        return 'Welcome'

    @staticmethod
    def load_dataframe():
        purchaseOrderRepository = PurchaseOrderRepository()
        df_purchase_order = purchaseOrderRepository.load_purchase_order_parquet()

        return df_purchase_order
    
    @staticmethod
    @app.route("/purchases", methods=["GET"])
    def get_purchases_order():
        """
        Get a purchase list
        ---
        tags:
           - Get all purchases
        responses:
            200:
                description: Success to get a purchase list.
            500:
                description: Failure to get a purchase list.
        """
        df_purchase_order = SemanticSearchTestApi().load_dataframe()

        return Response(
            df_purchase_order.head().to_json(orient="records"),
            mimetype="application/json"
        ), 200
    
    @staticmethod
    @app.route("/purchase/<string:purchase_order_number>", methods=["GET"])
    def get_purchase_order(purchase_order_number):
        """
        Get a purchase by ID
        ---
        parameters:
            - name: purchase_order_number
              in: path
              type: string
              required: true
              default: all
        tags:
           - Get a purchase order
        responses:
            200:
                description: Success to get a purchase order.
        """
        purchaseOrderRepository = PurchaseOrderRepository()
        df_purchase_order = purchaseOrderRepository.load_purchase_order_parquet()
        df_result = df_purchase_order.loc[
            df_purchase_order["purchase_order_number"] == purchase_order_number
        ]

        if len(df_result) == 0:
            return Response('{"result": "Record not found"}'), 404

        return Response(
            df_result.to_json(orient="records"),
            mimetype="application/json"
        ), 200
    
    @staticmethod
    @app.route("/purchase/item_name", methods=["POST"])
    def search_purchase_order():
        """
        Semantic Search - Endpoint to semantic search
        ---
        tags:
            - Semantic Search
        responses:
            200:
                description: Success to post a purchase order.
            500:
                description: Failure to post a purchase order.
        parameters:
            - name: body
              in: body
              required: true
        """
        json_query = json.loads(request.data)
        df_purchase_order = SemanticSearchTestApi().load_dataframe()
        df_purchase_order = df_purchase_order[
            df_purchase_order["item_name"] == json_query["sentence_query"]
        ]

        if len(df_purchase_order) == 0:
            return Response('{"result": "Record not found"}'), 404

        return Response(
            df_purchase_order.to_json(orient="records"),
            mimetype="application/json"
        ), 200


if __name__ == "__main__":
    app.run(debug=True)
