import os
import json
import base64

firebase_key_path = "/tmp/firebase_key.json"


def write_service_account_file():
    """
    Writes Firebase service account JSON from env variable to a file.
    """
    firebase_json_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64")

    if not firebase_json_b64:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_B64 env variable not set")

    decoded = base64.b64decode(firebase_json_b64)
    with open(firebase_key_path, "wb") as f:
        f.write(decoded)
