from sklearn.model_selection import train_test_split

from model import tune_model
from utils import PROCESSED_DATASET, TARGET_COL
from train import load_data

def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    tune_model(X_train, y_train)

if __name__ == '__main__':
    main()