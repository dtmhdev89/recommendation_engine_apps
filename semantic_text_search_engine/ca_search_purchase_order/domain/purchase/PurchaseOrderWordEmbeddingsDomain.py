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

            save_path = os.path.join("data", "purchase-order-data-2012-2015-.pkl")
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
