import shutil
from pathlib import Path

DATASET_TYPES = ['test', 'train']
DROP_COLS = ['Unnamed: 32', 'id']
TARGET_COL = 'diagnosis'
RAW_DATASET = 'raw_dataset/data.csv'
PROCESSED_DATASET = 'processed_dataset/data.csv'

def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)