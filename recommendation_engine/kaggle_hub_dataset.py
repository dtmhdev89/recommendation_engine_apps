import os
import shutil
import kagglehub


class DatasetDownloader():
    def __init__(self, download_dir=None) -> None:
        if download_dir is not None:
            self.download_dir = download_dir
        else:
            self.download_dir = os.path.join("./dataset")

    def download(self):
        os.makedirs(self.download_dir, exist_ok=True)

        path = kagglehub.dataset_download(
            "bhanupratapbiswas/fashion-products"
        )

        try:
            shutil.move(
                os.path.join(path, "fashion_products.csv"),
                self.download_dir
            )
        except FileNotFoundError:
            print("Error: Source file not found.")
        except shutil.Error as e:
            print("---", e)

        return self.download_dir
