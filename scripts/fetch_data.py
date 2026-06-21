import os
import gdown

def ensure_data_exists():
    """
    Checks if the dataset exists locally. If not, downloads it from Google Drive.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data", "processed")
    target_file = os.path.join(data_dir, "accepted_loans_final.csv")
    
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.exists(target_file):
        print(f"Dataset not found at {target_file}.")
        print("Downloading from Google Drive... This may take a while.")
        
        # TODO: Replace with the actual Google Drive File ID of your dataset
        # To get this ID, right click your file in Google Drive -> Share -> Copy Link
        # The link looks like: https://drive.google.com/file/d/<FILE_ID>/view
        DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
        
        # Note: The file on Google Drive must be set to "Anyone with the link can view"
        try:
            gdown.download(id=DRIVE_FILE_ID, output=target_file, quiet=False)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            print("Please ensure the DRIVE_FILE_ID is correct and the file is publicly shared.")
    else:
        print("Dataset already exists locally. Skipping download.")

if __name__ == "__main__":
    ensure_data_exists()
