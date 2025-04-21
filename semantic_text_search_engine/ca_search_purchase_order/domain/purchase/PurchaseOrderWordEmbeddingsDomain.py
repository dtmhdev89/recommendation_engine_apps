from domain.purchase.PurchaseOrderRepository import PurchaseOrderRepository
import os
import pickle
from sentence_transformers import SentenceTransformer

class PurchaseOrderWordEmbeddingsDomain:
    
    @staticmethod
    def create_word_embeddings_file():
        """Create serialized file"""
        try:
            purchaseOrderRepository = PurchaseOrderRepository()
            df = purchaseOrderRepository.load_purchase_order_parquet()
            df.reset_index(drop=True, inplace=True)
            model = SentenceTransformer('all-mpnet-base-v2')
            corpus_embeddings = model.encode(
                df["item_name_transformed"], convert_to_tensor=True
            )

            save_path = os.path.join(
                "data",
                "purchase-order-data-2012-2015-.pkl"
            )
            with open(save_path, "wb") as fOut:
                pickle.dump(
                    {'embeddings': corpus_embeddings},
                    fOut,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

            print("End process")
        except Exception as e:
            print("Error to generate word embeddings file: " + str(e))
            raise
    
    @staticmethod
    def load_word_embeddings_file_transformed():
        """Load serialized embeddings file"""
        try:
            save_path = os.path.join("data", "purchase-order-data-2012-2015-.pkl")
            with open(save_path, "rb") as fIn:
                stored_data = pickle.load(fIn)
                corpus_embeddings = stored_data["embeddings"]

                return corpus_embeddings
        except Exception as e:
            print("Error loading serialized Embeddings file: " + str(e))
            raise
    
    @staticmethod
    def transform_text_query_word_embeddings(query_embeddings):
        """Transforms search query text into Word Embeddings"""

        try:
            model = SentenceTransformer('all-mpnet-base-v2')
            query_embeddings = model.encode(
                query_embeddings,
                convert_to_tensor=True
            )

            return query_embeddings

        except Exception as e:
            print("Error transforming query text into Word Embeddings: " + str(e))
