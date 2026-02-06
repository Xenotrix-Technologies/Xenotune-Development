import os
import firebase_admin
from firebase_admin import credentials, storage

def init_firebase():
    if firebase_admin._apps:
        return

    key_path = os.getenv("FIREBASE_KEY_PATH")
    bucket_name = os.getenv("FIREBASE_BUCKET")

    if not key_path or not os.path.isfile(key_path):
        raise RuntimeError(f"Invalid Firebase key path: {key_path}")

    if not bucket_name:
        raise RuntimeError("FIREBASE_BUCKET not set")

    cred = credentials.Certificate(key_path)

    firebase_admin.initialize_app(cred, {
        "storageBucket":  os.getenv("FIREBASE_BUCKET")
    })

    print(f"✅ Firebase initialized with bucket: {bucket_name}")


def upload_to_firebase(local_file_path: str, firebase_path: str) -> str:
    init_firebase()

    if not os.path.isfile(local_file_path):
        raise FileNotFoundError(f"Local file not found: {local_file_path}")

    bucket = storage.bucket()
    blob = bucket.blob(firebase_path)

    blob.upload_from_filename(local_file_path)
    blob.make_public()

    return blob.public_url

if __name__ == "__main__":
    try:
        url = upload_to_firebase("output/music.mp3", "generated_music/music.mp3")
        print(f"Public URL: {url}")
    except Exception as err:
        print(f"Error: {err}")
