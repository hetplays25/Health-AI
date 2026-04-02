import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# LOAD DATASET
# ------------------------------
df = pd.read_csv("health_dataset.csv")

# Features & Target
X = df.drop("disease", axis=1)
y = df["disease"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------
# TRAIN MODEL
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# SAVE FILES
# ------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ model.pkl and label_encoder.pkl created successfully!")