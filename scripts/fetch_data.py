import os
#import gdown
#from dotenv import load_dotenv

#load_dotenv()


def ensure_data_exists():
    """
    Checks if the dataset exists locally. If not, downloads it from Google Drive.
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data", "processed")
    target_file = os.path.join(data_dir, "accepted_loans_final.csv")

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(target_file):
        print("Dataset already exists locally. Skipping download.")
        return

    print(f"Dataset not found at {target_file}.")
    # print("Downloading from Google Drive...")

    # drive_file_id = os.getenv("GOOGLE_DRIVE_FILE_ID")

    # if not drive_file_id:
    #     raise ValueError(
    #         "GOOGLE_DRIVE_FILE_ID is not set in the .env file."
    #     )

    # try:
    #     gdown.download(
    #         id=drive_file_id,
    #         output=target_file,
    #         quiet=False
    #     )

    #     if os.path.exists(target_file):
    #         print("Download complete!")
    #     else:
    #         print("Download appears to have failed.")

    # except Exception as e:
    #     print(f"Error downloading from Google Drive: {e}")
    #     print(
    #         "Make sure the file ID is correct and the file is shared "
    #         "as 'Anyone with the link can view'."
    #     )

def get_dataset_path():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(BASE_DIR, "..", "data", "processed", "accepted_loans_final.csv")


if __name__ == "__main__":
    ensure_data_exists()