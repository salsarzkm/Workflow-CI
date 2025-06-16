import mlflow
import argparse
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='predpersonal_preprocessing.csv')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop(columns='Personality')
y = df['Personality']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set experiment
mlflow.set_experiment("modelling-experiment")

# 💡 Langsung autolog saja, TANPA start_run()
mlflow.sklearn.autolog()

# Train model (mlflow run akan otomatis mencatat ke active run-nya)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Akurasi:", acc)
print("Laporan klasifikasi:\n", report)

# Simpan classification report
with open("classification_report.txt", "w") as f:
    f.write(report)
mlflow.log_artifact("classification_report.txt")
