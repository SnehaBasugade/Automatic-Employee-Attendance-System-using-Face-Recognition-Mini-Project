# Automatic Employee Attendance System using Face Recognition Mini Project



## 📌 Project Overview

This project is a **Face Recognition Based Employee Attendance System** developed for **Ishanvi Cranes Pvt Ltd, Palus**.

The system automatically identifies employees using facial recognition technology and records attendance with date and time. The solution eliminates manual attendance processes and improves accuracy, efficiency, and security.


## 🚀 Key Features

* Real-time face detection using webcam
* Face recognition using FaceNet embeddings
* MTCNN face detection
* SVM-based employee classification
* Automatic attendance marking
* Employee database management
* Attendance report generation
* Real-time attendance tracking
* High recognition accuracy (>95%)


## 🛠 Technologies Used

### Programming Language

* Python

### Computer Vision

* OpenCV
* MTCNN

### Deep Learning

* FaceNet

### Machine Learning

* Scikit-Learn (SVM)

### Database

* MySQL

### GUI

* Tkinter

### Data Processing

* Pandas
* NumPy

### Model Storage

* Joblib


## 📂 Project Structure

```text
Employee-Attendance-System/
│
├── dataset/
├── embeddings/
├── models/
│
├── attendance.py
├── FaceCapture.py
├── generate_embeddings.py
├── train_model.py
├── evaluate_model.py
├── standalone_attendance.py
│
├── Person_Info_New.csv
├── requirements.txt
│
└── README.md


🔄 Project Workflow

Step 1: Face Data Collection

Employee face images are captured using a webcam and stored in the dataset folder.


Webcam → Face Capture → Dataset

Step 2: Face Embedding Generation

FaceNet extracts facial features and generates embeddings for each employee image.
Dataset → FaceNet → Embeddings


Step 3: Model Training

Generated embeddings are used to train an SVM classifier.
Embeddings → SVM Training → Trained Model


Step 4: Employee Recognition

Live webcam feed is processed using:

* MTCNN Face Detection
* FaceNet Feature Extraction
* SVM Classification

Camera → Detection → Recognition


Step 5: Attendance Marking

If the employee is recognized:

* Employee Name
* Employee ID
* Date
* Time

are automatically recorded.


📋 Execution Flow

Capture Employee Images

python FaceCapture.py

Generate Face Embeddings


python generate_embeddings.py


Train Recognition Model

python train_model.py

Evaluate Model

python evaluate_model.py

Run Attendance System


python attendance.py

📊 Results

* Recognition Accuracy: >95%
* Automatic Attendance Recording
* Reduced Manual Effort
* Faster Employee Verification
* Improved Attendance Management



🏢 Industrial Application

This system can be deployed in:

* Manufacturing Industries
* Corporate Offices
* Educational Institutions
* Hospitals
* Government Organizations


🎯 Future Enhancements

* Cloud Database Integration
* Face Anti-Spoofing Detection
* Employee Dashboard
* Power BI Analytics Integration
* AES Encryption for Secure Attendance Records


👩💻 Developer

Sneha Basugade

B.Tech Computer Science and Engineering
(Internet of Things and Cyber Security including Blockchain Technology)

Project Sponsor: Ishanvi Cranes Pvt Ltd, Palus


📜 License

This project is developed for educational and industrial demonstration purposes.
