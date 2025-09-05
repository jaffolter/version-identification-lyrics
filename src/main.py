# ------------------------------------------------------------
# Main script called when running 'make run'
# You can modify to call any other functions or modules
# ------------------------------------------------------------
from pathlib import Path

from livi.config import settings


def main():
    print("Hello from main script!")

    # Retrieve global variables from config.settings
    data_path = Path(settings.DATA_DIR)
    audio_path = Path(settings.AUDIO_DIR)
    embeddings_path = Path(settings.EMBEDDINGS_DIR)

    print(data_path)
    print(audio_path)
    print(embeddings_path)

    # call functions or modules here

    # mymodule.run_stuff()


if __name__ == "__main__":
    main()
