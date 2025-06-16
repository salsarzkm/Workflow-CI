import mlflow
import argparse
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set tracking URI 
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("file:./mlruns")  # tanpa direktori Windows

# Argparse agar MLflow bisa pass --data_path
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='predpersonal_preprocessing.csv')
args = parser.parse_args()

# Load dataset dari parameter
df = pd.read_csv(args.data_path)

# Memisahkan target dan fitur
X = df.drop(columns='Personality')  # kolom fitur
y = df['Personality']               # kolom target

# Bagi data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set tracking ke direktori lokal di proyek ini
# mlflow.set_tracking_uri("file:///C:/Users/Windows 10/Studpen_Msml/Workflow-CI/MLProject/mlruns")

# Experiment bernama 'modelling-experiment' (otomatis dibuat kalau belum ada)
mlflow.set_experiment("modelling-experiment")

# Start run & log
run = mlflow.start_run() if mlflow.active_run() is None else mlflow.active_run()

# Autolog *setelah* start_run
mlflow.sklearn.autolog()