import shutil
import zipfile
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

COMPETITION = 'rossmann-store-sales'
FILES = ('train.csv', 'store.csv')
RAW_DIR = Path(__file__).resolve().parent


def download(name: str) -> Path:
    downloaded = Path(
        kagglehub.competition_download(
            COMPETITION,
            path=name,
            output_dir=str(RAW_DIR),
            force_download=True,
        )
    )
    if zipfile.is_zipfile(downloaded):
        with zipfile.ZipFile(downloaded) as archive:
            archive.extract(name, RAW_DIR)
        downloaded.unlink()
    return RAW_DIR / name


def cleanup() -> None:
    keep = {*FILES, Path(__file__).name, '.gitkeep'}
    for item in RAW_DIR.iterdir():
        if item.name in keep:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


if __name__ == '__main__':
    load_dotenv()
    try:
        for name in FILES:
            print(download(name))
    finally:
        cleanup()
