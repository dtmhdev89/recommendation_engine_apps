import os
import shutil
import kagglehub


class DatasetDownloader():
    def __init__(
        self,
        dataset_id,
        download_dir=None,
        keep_cache_dir=True
    ) -> None:
        self.dataset_id = dataset_id
        self.keep_cache_dir = keep_cache_dir

        if download_dir is not None:
            self.download_dir = download_dir
        else:
            self.download_dir = os.path.join("./dataset")

    def download(self):
        os.makedirs(self.download_dir, exist_ok=True)
        path = kagglehub.dataset_download(self.dataset_id)

        try:
            filename_list = [
                f_name for f_name in os.listdir(path) if f_name.endswith(".csv")
            ]
            for f_name in filename_list:
                shutil.move(
                    os.path.join(path, f_name),
                    self.download_dir
                )

            if not self.keep_cache_dir:
                shutil.rmtree(path)

        except FileNotFoundError:
            print("Error: Source file not found.")
        except shutil.Error as e:
            print("---", e)

        return self.download_dir
