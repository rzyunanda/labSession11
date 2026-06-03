"""
Training model prediksi harga rumah menggunakan dataset California Housing dari sklearn.
Model disimpan ke house_model.pkl
"""
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def main():
    # 1. Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target  # target dalam ratusan ribu USD (mis. 4.5 = $450.000)
    feature_names = list(data.feature_names)

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    pred = model.predict(X_test)
    print(f"R2 Score : {r2_score(y_test, pred):.4f}")
    print(f"MAE      : {mean_absolute_error(y_test, pred):.4f}")

    # 5. Simpan model + nama fitur ke pkl
    with open("house_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feature_names}, f)

    print("Model tersimpan ke house_model.pkl")
    print("Fitur:", feature_names)


if __name__ == "__main__":
    main()
