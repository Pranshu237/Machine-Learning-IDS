import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import DATA_PATH, RANDOM_STATE, TEST_SIZE

LABEL_COL = "Label"


def load_clean(path=DATA_PATH):
    """Read the merged CICIDS-2017 CSV and return clean features and string labels."""
    df = pd.read_csv(path, na_values=["Infinity", "NaN", "inf", "-inf", "INF"], low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop redundant duplicated column if present
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])

    # Clean label values (fix corrupted encoding characters)
    labels = df[LABEL_COL].astype(str).str.strip()
    labels = labels.str.replace("\x96", "-", regex=False)
    labels = labels.str.replace("�", "-", regex=False)
    labels = labels.str.replace("–", "-", regex=False)
    labels = labels.str.replace(r"Web Attack\s+-?\s*", "Web Attack - ", regex=True)
    df[LABEL_COL] = labels

    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL]).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    del df

    # Drop rows with missing values, then exact duplicates
    data = pd.concat([X, y], axis=1).dropna().drop_duplicates()
    del X, y

    # float32 halves memory use. Duplicates are removed first, at full
    # precision, so the row count does not depend on this. (scikit-learn's
    # tree models convert their input to float32 anyway.)
    labels = data[LABEL_COL]
    features = data.drop(columns=[LABEL_COL]).astype(np.float32)
    return features, labels


def preprocess(path=DATA_PATH):
    X, y = load_clean(path)

    le = LabelEncoder()
    y = le.fit_transform(y)
    X = X.to_numpy()

    # Split first, then fit the scaler on the training set only,
    # so nothing about the test set leaks into training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le.classes_
