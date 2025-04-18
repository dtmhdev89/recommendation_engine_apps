import os
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
    
    def set_dataframe_index(self):
        """Define the index of the dataframe"""
        self.df_purchase_order.index += 1

    def delete_columns(self):
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

            self.df_purchase_order.drop(list_remove_columns, axis=1, inplace=True)
        except Exception as e:
            print("Error deleting data columns: " + str(e))
            raise
    
    def rename_columns_name(self):
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
            self.df_purchase_order.rename(
                columns=dict_new_columns_name,
                inplace=True
            )
        except Exception as e:
            print("Error renaming columns: " + str(e))
            raise
    
    def removing_missing_values(self):
        """Removing missing values"""
        try:
            self.df_purchase_order.dropna(subset=['item_name'], inplace=True)
        except Exception as e:
            print("Error removing missing values: " + str(e))
            raise

    def removing_anomalies(self):
        """Removing anomalies"""
        try:
            self.df_purchase_order = self.df_purchase_order[self.df_purchase_order["item_name"] != 'Discount']
        except Exception as e:
            print("Error removing anomalies: " + str(e))
            raise
    
    def delete_stopwords(self):
        """Delete stopwords"""
        try:
            stop = stopwords.words("english")
            self.df_purchase_order['item_name_transformed'] = self.df_purchase_order['item_name'].apply(
                lambda x: ' '.join(
                    [word for word in x.split() if word not in stop]
                )
            )
        except Exception as e:
            print("Error to delete stopwords: " + str(e))
            raise
    
    def text_lemmatize(self):
        """Lemmatize words"""
        try:
            spacy_nlp = spacy.load("en_core_web_lg")
            self.df_purchase_order['item_name_transformed'] = self.df_purchase_order.apply(
                lambda row: " ".join(
                    [w.lemma_ for w in spacy_nlp(row)]
                )
            )

        except Exception as e:
            print("Error to lemmatize words: " + str(e))
            raise
    
    def data_capitalization(self):
        """Convert text to lowercase"""
        try:
            self.df_purchase_order["item_name_transformed"] = self.df_purchase_order['item_name'].str.lower()
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
                self.set_dataframe_index()
                self.delete_columns()
                self.rename_columns_name()
            
            self.removing_missing_values()
            self.removing_anomalies()
            self.delete_stopwords()
            self.text_lemmatize()
            self.data_capitalization()

        except Exception as e:
            print("Error preprocessing dataframe: " + str(e))
            raise


# purchaseOrderPreprocessingDomain = PurchaseOrderPreprocessingDomain()
# purchaseOrderPreprocessingDomain.data_preprocessing()
