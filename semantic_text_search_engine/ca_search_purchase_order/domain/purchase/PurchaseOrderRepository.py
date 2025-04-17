import pandas as pd
import os


class PurchaseOrderRepository:
    """Class for data persistence"""

    @staticmethod
    def load_purchase_order_csv():
        """Load data file in csv format"""

        try:
            data_path = os.path.join(
                "../../data/", "purchase-order-data-2012-2015-.parquet"
            )
            print(data_path)
            return pd.read_parquet(
                data_path,
                delimiter=',',
                encoding="utf-8"
            )
        except Exception as e:
            print("Error loading data file: " + e.__str__())
            raise


# module test
purchaseOrderRepository = PurchaseOrderRepository()
print(purchaseOrderRepository.load_purchase_order_csv().head())
