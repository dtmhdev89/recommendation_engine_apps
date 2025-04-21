import pandas as pd
from domain.purchase.PurchaseOrderPreprocessingDomain import PurchaseOrderPreprocessingDomain
from domain.purchase.PurchaseOrderRepository import PurchaseOrderRepository
from domain.purchase.PurchaseOrderWordEmbeddingsDomain import PurchaseOrderWordEmbeddingsDomain
from sentence_transformers import SentenceTransformer, util


class PurchaseOrderDomain:
    """Domain Class"""

    @staticmethod
    def semantic_search(sentence_query):
        try:
            # pre-processing of the sentence sent as search criteria
            purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
            sentence_transformed = purchaseOrderPreprocessingDomain.text_query_preprocessing(sentence_query)

            purchaseOrderWordEmbeddingsDomain = PurchaseOrderWordEmbeddingsDomain()
            sentence_query_embeddings = purchaseOrderWordEmbeddingsDomain.transform_text_query_word_embeddings(sentence_transformed)

            # load embeddings file
            corpus_embeddings = purchaseOrderWordEmbeddingsDomain.load_word_embeddings_file_transformed()

            total_result_lines = 200
            list_result_query = util.semantic_search(
                sentence_query_embeddings,
                corpus_embeddings,
                top_k=total_result_lines
            )
            list_result_query = list_result_query[0]

            purchaseOrderRepository = PurchaseOrderRepository()
            df_purchase_order = purchaseOrderRepository.load_purchase_order_parquet()
            df_purchase_order.reset_index(drop=True, inplace=True)

            # Create dataframe to display search results
            df_search_result = pd.DataFrame(list_result_query)
            df_search_result = df_search_result.set_index('corpus_id')

            df_search_result['score'] = df_search_result['score'].round(decimals=2) * 100
            df_search_result['creation_date'] = df_purchase_order['creation_date']
            df_search_result['purchase_order_number'] = df_purchase_order['purchase_order_number']
            df_search_result['department_name'] = df_purchase_order['department_name']
            df_search_result['supplier_name'] = df_purchase_order['supplier_name']
            df_search_result['item_name'] = df_purchase_order['item_name']
            df_search_result['item_description'] = df_purchase_order['item_description']
            df_search_result['quantity'] = df_purchase_order['quantity']
            df_search_result['unit_price'] = df_purchase_order['unit_price']
            df_search_result['total_price'] = df_purchase_order['total_price']
            df_search_result['class'] = df_purchase_order['class']
            df_search_result['class_title'] = df_purchase_order['class_title']
            df_search_result['item_name_transformed'] = df_purchase_order['item_name_transformed']

            return df_search_result[
                [
                    'score', 'purchase_order_number', 'item_name',
                    'item_name_transformed', 'quantity', 'total_price']]

        except Exception as e:
            print("Semantic search error: " + str(e))
            raise


embeddings = PurchaseOrderWordEmbeddingsDomain()
embeddings.create_word_embeddings_file()
purchaseOrderDomain = PurchaseOrderDomain()
print(purchaseOrderDomain.semantic_search("agency"))
