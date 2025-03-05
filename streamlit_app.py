import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Fungsi untuk memuat data dari file Excel
def load_data():
    uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx", "xls"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        return data
    else:
        return None

# Fungsi untuk mengonversi nilai numerik menjadi kategori
def convert_to_category(data, threshold):
    # Jika final_grade berupa numerik, ubah menjadi kategori
    if data['final_grade'].dtype in ['int64', 'float64']:
        data['final_grade'] = data['final_grade'].apply(lambda x: 'Lulus' if x > threshold else 'Tidak Lulus')
    return data

# Fungsi pra-pemrosesan data
def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])  # Fitur
    y = data[target_column]  # Target
    # Pembagian data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standarisasi data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Fungsi untuk regresi (prediksi nilai)
def regress_data(X_train, X_test, y_train, y_test, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "svm":
        model = SVR()
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=5)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Fungsi untuk klasifikasi (prediksi kategori)
def classify_data(X_train, X_test, y_train, y_test, model_type="random_forest"):
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        model = SVC()
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

# Fungsi untuk visualisasi hasil
def plot_results(predictions, y_test, model_type):
    plt.figure(figsize=(10,6))
    if model_type in ["regression", "svm", "knn"]:  # Model regresi
        plt.scatter(y_test, predictions)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Prediction vs Actual Values")
    else:  # Model klasifikasi
        plt.scatter(y_test, predictions)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Classification Predictions vs Actual Values")
    plt.show()

# Wrapper utama untuk analisis
def analyze_data(model_type="regression"):
    # Memuat file data
    data = load_data()
    if data is None:
        st.warning("Silakan upload file Excel terlebih dahulu.")
        return

    # Mengambil nama kolom target dari pengguna
    target_column = st.text_input("Masukkan nama kolom target:", "final_grade")

    # Meminta pengguna untuk memasukkan threshold untuk kategori
    threshold = st.number_input("Masukkan threshold untuk kategori (misalnya 70 untuk Lulus/Tidak Lulus):", min_value=0, max_value=100, value=70)

    # Mengonversi final_grade ke kategori jika berupa numerik
    data = convert_to_category(data, threshold)

    if target_column:
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column)

        if model_type == "regression":
            model_choice = st.selectbox("Pilih model untuk regresi:", ["random_forest", "svm", "knn"])
            mse, predictions = regress_data(X_train, X_test, y_train, y_test, model_choice)
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"Predictions vs Actual: {list(zip(predictions, y_test))}")
            st.pyplot(plt.gcf())
        elif model_type == "classification":
            model_choice = st.selectbox("Pilih model untuk klasifikasi:", ["random_forest", "svm", "knn"])
            accuracy, predictions = classify_data(X_train, X_test, y_train, y_test, model_choice)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Predictions vs Actual: {list(zip(predictions, y_test))}")
            st.pyplot(plt.gcf())

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    st.title("AI Wrapper untuk Analisis Pendidikan")
    model_type = st.selectbox("Pilih jenis model:", ["regression", "classification"])
    analyze_data(model_type)
