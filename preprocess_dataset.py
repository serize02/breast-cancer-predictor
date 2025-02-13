from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import (
    DROP_COLS,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COL
)

def read_dataset(
    filename: str, drop_cols: List[str], target_column: str
) -> pd.DataFrame:
    df = pd.read_csv(filename).drop(columns=drop_cols)
    df[target_column] = LabelEncoder().fit_transform(df[target_column])
    return df

def scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df_features.values)
    return pd.DataFrame(X, columns=df_features.columns)

def main():
    
    data = read_dataset(filename=RAW_DATASET, drop_cols=DROP_COLS, target_column=TARGET_COL)
    
    data_features = scale_data(
        data.drop(columns=TARGET_COL, axis=1)
    )

    data_labels = data[TARGET_COL]
    data = pd.concat([data_features, data_labels], axis=1)
    data.to_csv(PROCESSED_DATASET, index=None)

if __name__ == '__main__':
    main()
