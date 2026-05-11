import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import preprocess_data, scale_data

def train_model(file_path, target_column, model_name):
    df = pd.read_csv(file_path)

    if "child_illiteracy" in file_path:
        df['High_Illiteracy'] = df['est_illiterate_children_millions'] > df['est_illiterate_children_millions'].mean()
        df['High_Illiteracy'] = df['High_Illiteracy'].astype(int)
        target_column = "High_Illiteracy"
        df = df.drop("est_illiterate_children_millions", axis=1)

    elif "shopping" in file_path:
        df['High_Spender'] = df['Purchase Amount (USD)'] > df['Purchase Amount (USD)'].mean()
        df['High_Spender'] = df['High_Spender'].astype(int)
        target_column = "High_Spender"
        df = df.drop("Purchase Amount (USD)", axis=1)

    elif "sp500" in df.columns:
        df['Market_Drop'] = df['sp500'].diff() < 0
        df['Market_Drop'] = df['Market_Drop'].astype(int)
        df = df.dropna()
        target_column = "Market_Drop"

    df = preprocess_data(df)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test, scaler = scale_data(X_train, X_test)

    dt = DecisionTreeClassifier().fit(X_train, y_train)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    nn = MLPClassifier(max_iter=500).fit(X_train, y_train)

    joblib.dump({
        "dt": dt,
        "knn": knn,
        "nn": nn,
        "scaler": scaler,
        "columns": X.columns.tolist()
    }, f"models/{model_name}.pkl")

    print(f"{model_name} trained successfully")

if __name__ == "__main__":
    train_model("data/Healthcare-Diabetes.csv", "Outcome", "diabetes_models")
    train_model("data/child_illiteracy_by_country.csv", "", "child_illiteracy_models")
    train_model("data/customer_shopping_behavior.csv", "", "shopping_models")
    train_model("data/market_reaction.csv", "", "trade_models")