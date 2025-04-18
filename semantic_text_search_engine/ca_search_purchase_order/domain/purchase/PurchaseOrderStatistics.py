from domain.purchase.PurchaseOrderRepository import PurchaseOrderRepository


class PurchaseOrderStatistics:
    """Statistic Class"""

    def __init__(self) -> None:
        self.purchaseOrderRepository = PurchaseOrderRepository()
        self.df_purchase_order = self.purchaseOrderRepository.load_purchase_order_parquet()

    def view_statistics(self):
        """View Statistics"""

        try:
            print("----> Dataset Purchase Order")
            print(self.df_purchase_order.head())

            print("\n----> Total rows Dataset Purchase Order")
            print(self.df_purchase_order.shape[0])

            print("\n----> Columns Dataset Purchase Order:")
            print(self.df_purchase_order.columns)

            print("\n----> Describing Dataset Purchase Order:")
            print(self.df_purchase_order.describe())

            print("\n----> Describing Dataset Purchase Order:")
            print(self.df_purchase_order["department_name"].unique())

        except Exception as e:
            print("Error to show statistics: " + str(e))


PurchaseOrderStatistics = PurchaseOrderStatistics()
PurchaseOrderStatistics.view_statistics()
