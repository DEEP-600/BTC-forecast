import sqlite3
import shutil
import os

REPO_DB = os.path.join(os.path.dirname(__file__), "predictions_seed.db")
VOLUME_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data")
VOLUME_DB = os.path.join(VOLUME_DIR, "predictions.db")


def seed():
    # Only seed if volume db doesn't exist yet
    if not os.path.exists(VOLUME_DB):
        os.makedirs(VOLUME_DIR, exist_ok=True)
        if os.path.exists(REPO_DB):
            shutil.copy2(REPO_DB, VOLUME_DB)
            print("Seeded predictions.db from repo backup")
        else:
            print("No seed db found, starting fresh")
    else:
        print("Volume DB already exists, skipping seed")


if __name__ == "__main__":
    seed()
