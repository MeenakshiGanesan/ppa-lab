import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import preprocess_data

def evaluate_model(file_path, model_name):
    df = pd.read_csv(file_path)

    if "child_illiteracy" in file_path:
        df['High_Illiteracy'] = df['est_illiterate_children_millions'] > df['est_illiterate_children_millions'].mean()
        df['High_Illiteracy'] = df['High_Illiteracy'].astype(int)
        df = df.drop("est_illiterate_children_millions", axis=1)  
        target_column = "High_Illiteracy"

    elif "shopping" in file_path:
        df['High_Spender'] = df['Purchase Amount (USD)'] > df['Purchase Amount (USD)'].mean()
        df['High_Spender'] = df['High_Spender'].astype(int)
        df = df.drop("Purchase Amount (USD)", axis=1)  
        target_column = "High_Spender"

    elif "sp500" in df.columns:
        df['Market_Drop'] = df['sp500'].diff() < 0
        df['Market_Drop'] = df['Market_Drop'].astype(int)
        df = df.dropna()
        target_column = "Market_Drop"

    else:
        target_column = "Outcome"  

    df = preprocess_data(df)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = joblib.load(f"models/{model_name}.pkl")

    dt = models["dt"]
    knn = models["knn"]
    nn = models["nn"]
    scaler = models["scaler"]    
    required_cols = models["columns"]

    X_test = X_test.reindex(columns=required_cols, fill_value=0)
    X_test = scaler.transform(X_test)

    dt_pred = dt.predict(X_test)
    knn_pred = knn.predict(X_test)
    nn_pred = nn.predict(X_test)

    return {
        "Decision Tree": accuracy_score(y_test, dt_pred),
        "KNN": accuracy_score(y_test, knn_pred),
        "Neural Network": accuracy_score(y_test, nn_pred)
    }

DATASETS = {
    "Diabetes": ("data/Healthcare-Diabetes.csv", "diabetes_models"),
    "Illiteracy": ("data/child_illiteracy_by_country.csv", "child_illiteracy_models"),
    "Shopping": ("data/customer_shopping_behavior.csv", "shopping_models"),
    "Trade": ("data/market_reaction.csv", "trade_models")
}

if __name__ == "__main__":

    results = {}
    for name, (path, model) in DATASETS.items():
        results[name] = evaluate_model(path, model)

    print("\nFINAL MODEL COMPARISON")

    for dataset, scores in results.items():
        print(f"\n{dataset}")
        for model, score in scores.items():
            print(f"{model}: {score:.4f}")

    df_results = pd.DataFrame(results).T
    print("\nSummary Table:\n")
    print(df_results)

def get_all_results():
    results = {}

    for name, (path, model) in DATASETS.items():
        results[name] = evaluate_model(path, model)

    return results