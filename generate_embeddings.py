import pandas as pd
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

# Create embeddings folder if not exists
os.makedirs("embeddings", exist_ok=True)

# Load CSV
df = pd.read_csv("Person_Info_New.csv")

# Initialize detectors
detector = MTCNN()
embedder = FaceNet()

embeddings = []
names = []
employee_ids = []

for index, row in df.iterrows():

    image_path = row['Image_path']
    name = row['Name']
    employee_id = row['Employee_ID']

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found:", image_path)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(img)

    if len(faces) == 0:
        print("No face detected:", image_path)
        continue

    # Take first detected face
    x, y, w, h = faces[0]['box']

    # Fix negative values
    x, y = abs(x), abs(y)

    face = img[y:y+h, x:x+w]

    # Resize for FaceNet
    face = cv2.resize(face, (160,160))

    # Generate embedding
    embedding = embedder.embeddings([face])[0]

    embeddings.append(embedding)
    names.append(name)
    employee_ids.append(employee_id)

# Convert to numpy
embeddings = np.array(embeddings)
names = np.array(names)
employee_ids = np.array(employee_ids)

# Save embeddings
np.savez(
    "embeddings/face_embeddings.npz",
    embeddings=embeddings,
    names=names,
    employee_ids=employee_ids
)

print("Embeddings Generated Successfully")
print("Total embeddings created:", len(embeddings))