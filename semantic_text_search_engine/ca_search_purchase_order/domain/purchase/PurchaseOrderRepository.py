import pandas as pd
import os


def set_pd_display_option():
    """Set pandas display options"""

    pd.set_option("display.width", 600)
    pd.set_option("display.max_columns", 20)


class PurchaseOrderRepository:
    """Class for data persistence"""

    @staticmethod
    def load_purchase_order_csv():
        """Load data file in csv format"""

        try:
            data_path = os.path.join(
                "data",
                "purchase-order-data-2012-2015-.csv"
            )

            return pd.read_csv(
                data_path,
                delimiter=',',
                encoding="utf-8"
            )
        except Exception as e:
            print("Error loading data file: " + e.__str__())
            raise
    
    @staticmethod
    def load_purchase_order_parquet():
        try:
            data_path = os.path.join(
                "data",
                "purchase-order-data-2012-2015-.parquet"
            )

            return pd.read_parquet(
                data_path
            )
        except Exception as e:
            print("Error loading data file: " + e.__str__())
            raise


set_pd_display_option()
# module test
# purchaseOrderRepository = PurchaseOrderRepository()
# print(purchaseOrderRepository.load_purchase_order_csv().head())
