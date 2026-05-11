import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()
    drop_cols = []

    for col in df.columns:
        if "id" in col.lower():
            drop_cols.append(col)

    df = df.drop(columns=drop_cols, errors='ignore')

    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler