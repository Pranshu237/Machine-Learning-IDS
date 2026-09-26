import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def preprocess():
    # Read CSV with string infinity handling
    df = pd.read_csv(DATA_PATH, na_values=['Infinity', 'NaN', 'inf', '-inf', 'INF'], low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop redundant duplicated column if present
    if 'Fwd Header Length.1' in df.columns:
        df = df.drop(columns=['Fwd Header Length.1'])

    # Clean label values (fix corrupted encoding characters)
    label_col = 'Label'
    df[label_col] = df[label_col].astype(str).str.strip()
    df[label_col] = df[label_col].replace({
        'Web Attack  Brute Force': 'Web Attack - Brute Force',
        'Web Attack  XSS': 'Web Attack - XSS',
        'Web Attack  Sql Injection': 'Web Attack - Sql Injection',
        'Web Attack \ufffd Brute Force': 'Web Attack - Brute Force',
        'Web Attack \ufffd XSS': 'Web Attack - XSS',
        'Web Attack \ufffd Sql Injection': 'Web Attack - Sql Injection',
    })
    df[label_col] = df[label_col].str.replace('\x96', '-', regex=False).str.replace('\ufffd', '-', regex=False).str.replace('–', '-', regex=False)

    # Separate features and label
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # Convert all feature columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Remove infinite values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Combine back to drop rows with missing values
    data = pd.concat([X, y], axis=1).dropna().drop_duplicates()

    y = data[label_col]
    X = data.drop(columns=[label_col])

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, le.classes_