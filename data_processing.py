import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Debugging: Cek kolom dalam dataset
    print("Kolom dataset:", data.columns.tolist())
    
    # Normalisasi nama kolom (opsional)
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    
    # Categorical columns
    categorical_columns = ['Jenis_Kelamin', 'Pendidikan', 'Status_Ekonomi', 'Pekerjaan', 'Penggunaan_KB', 'Sumber_Informasi']
    
    # Pastikan kolom ada di dataset
    for col in categorical_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset.")
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Prepare X and y
    X = data[categorical_columns]
    y = data['target']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Return processed data and mappings
    feature_mapping = {col: label_encoders[col].classes_.tolist() for col in categorical_columns}
    return X_train, X_test, y_train, y_test, feature_mapping
