import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import pandas as pd

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load embeddings
data = np.load("embeddings/face_embeddings.npz")

X = data['embeddings']
names = data['names']
employee_ids = data['employee_ids']

# Encode names
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(names)

# Train SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# Create mapping between name and student_id
mapping_df = pd.DataFrame({
    "Name": names,
    "Employee_ID": employee_ids
}).drop_duplicates()

# Save files
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")
mapping_df.to_csv("models/employee_mapping.csv", index=False)

print("Model trained successfully")
print("Total classes:", len(set(names)))