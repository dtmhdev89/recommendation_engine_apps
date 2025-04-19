import os
import pandas as pd
from domain.purchase.PurchaseOrderRepository import PurchaseOrderRepository
from nltk.corpus import stopwords
import spacy
import spacy.cli
spacy.cli.download("en_core_web_lg")


class PurchaseOrderPreprocessingDomain:
    """Class for preprocessing and clean the dataset"""

    def __init__(self, csv_load=False) -> None:
        self.csv_load = csv_load
        self.purchaseOrderRepository = PurchaseOrderRepository()
        if csv_load:
            self.df_purchase_order = self.purchaseOrderRepository.load_purchase_order_csv()
        else:
            self.df_purchase_order = self.purchaseOrderRepository.load_purchase_order_parquet()
    
    @staticmethod
    def set_dataframe_index(df):
        """Define the index of the dataframe"""
        df.index += 1

        return df

    @staticmethod
    def delete_columns(df):
        """Delete unused columns"""

        try:
            list_remove_columns = [
                "Purchase Date",
                "Fiscal Year",
                "LPA Number",
                "Requisition Number",
                "Acquisition Method",
                "Sub-Acquisition Method",
                "Supplier Code",
                "Supplier Qualifications",
                "Supplier Zip Code",
                "Acquisition Type",
                "Sub-Acquisition Type",
                "CalCard",
                "Classification Codes",
                'Normalized UNSPSC',
                'Commodity Title',
                'Segment',
                'Segment Title',
                'Location',
                'Family',
                'Family Title',
                'REMOVE AMERISOURCE'
            ]

            df.drop(list_remove_columns, axis=1, inplace=True)
            
            return df
        except Exception as e:
            print("Error deleting data columns: " + str(e))
            raise
    
    @staticmethod
    def rename_columns_name(df):
        """Rename columns name"""
        try:
            dict_new_columns_name = {
                "Creation Date": "creation_date",
                "Purchase Order Number": "purchase_order_number",
                "Department Name": "department_name",
                "Supplier Name": "supplier_name",
                "Item Name": "item_name",
                "Item Description": "item_description",
                "Quantity": "quantity",
                "Unit Price": "unit_price",
                "Total Price": "total_price",
                "Class": "class",
                "Class Title": "class_title",
            }
            df.rename(
                columns=dict_new_columns_name,
                inplace=True
            )

            return df
        except Exception as e:
            print("Error renaming columns: " + str(e))
            raise
    
    @staticmethod
    def removing_missing_values(df):
        """Removing missing values"""
        try:
            df.dropna(subset=['item_name'], inplace=True)

            return df
        except Exception as e:
            print("Error removing missing values: " + str(e))
            raise

    @staticmethod
    def removing_anomalies(df):
        """Removing anomalies"""
        try:
            df = df[df["item_name"] != 'Discount']

            return df
        except Exception as e:
            print("Error removing anomalies: " + str(e))
            raise
    
    @staticmethod
    def delete_stopwords(df):
        """Delete stopwords"""
        try:
            stop = stopwords.words("english")
            df['item_name_transformed'] = df['item_name'].apply(
                lambda x: ' '.join(
                    [word for word in x.split() if word not in stop]
                )
            )

            return df
        except Exception as e:
            print("Error to delete stopwords: " + str(e))
            raise
    
    @staticmethod
    def text_lemmatize(df):
        """Lemmatize words"""
        try:
            spacy_nlp = spacy.load("en_core_web_lg")
            df['item_name_transformed'] = df['item_name_transformed'].apply(
                lambda row: " ".join(
                    [w.lemma_ for w in spacy_nlp(row)]
                )
            )

            return df
        except Exception as e:
            print("Error to lemmatize words: " + str(e))
            raise
    
    @staticmethod
    def data_capitalization(df):
        """Convert text to lowercase"""
        try:
            df["item_name_transformed"] = df['item_name_transformed'].str.lower()

            return df
        except Exception as e:
            print("Error to text capitalization: " + str(e))
            raise
    
    @staticmethod
    def convert_df_parquet(df):
        """Convert dataframe file to parquet"""
        try:
            data_path = os.path.join(
                "data",
                "purchase-order-data-2012-2015-.parquet"
            )
            df.to_parquet(data_path, compression="brotli")
        except Exception as e:
            print("Error converting dataframe to Parquet file: " + str(e))
            raise

    def data_preprocessing(self):
        """Preprocessing dataframe"""
        try:
            if self.csv_load:
                self.df_purchase_order = self.set_dataframe_index(self.df_purchase_order)
                self.df_purchase_order = self.delete_columns(self.df_purchase_order)
                self.df_purchase_order = self.rename_columns_name(self.df_purchase_order)
            
            self.df_purchase_order = self.removing_missing_values(self.df_purchase_order)
            self.df_purchase_order = self.removing_anomalies(self.df_purchase_order)
            self.df_purchase_order = self.delete_stopwords(self.df_purchase_order)
            self.df_purchase_order = self.text_lemmatize(self.df_purchase_order)
            self.df_purchase_order = self.data_capitalization(self.df_purchase_order)

        except Exception as e:
            print("Error preprocessing dataframe: " + str(e))
            raise

    def text_query_preprocessing(self, text_query):
        """Preprocessing text query"""
        try:
            dict_query = {'item_name': [text_query]}
            df_query = pd.DataFrame(dict_query)
            df_query = self.removing_anomalies(df_query)
            df_query = self.delete_stopwords(df_query)
            df_query = self.text_lemmatize(df_query)
            df_query = self.data_capitalization(df_query)
            result_text_query = df_query.item_name_transformed.loc[0]
            print(result_text_query)
            return result_text_query
        except Exception as e:
            print("Error preprocessing text query: " + str(e))
            raise


purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
# purchaseOrderPreprocessingDomain.data_preprocessing()
purchaseOrderPreprocessingDomain.text_query_preprocessing("Ordering diapers")
