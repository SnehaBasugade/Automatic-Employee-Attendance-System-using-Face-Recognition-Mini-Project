import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import joblib
import pandas as pd
import datetime
import mysql.connector
import time
import sys

# Silence Keras prints for a clean terminal
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

print("====================================")
print("Loading Face Recognition Models...")
with HiddenPrints():
    detector = MTCNN()
    embedder = FaceNet()
svm_model = joblib.load("models/svm_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")
print("Models loaded successfully. ✅")
print("====================================")

threshold = 0.60
mapping_csv = "models/employee_mapping.csv"

def get_employee_map():
    if os.path.exists(mapping_csv):
        return pd.read_csv(mapping_csv)
    return pd.DataFrame(columns=["Employee_ID","Name"])

def get_db_connection():
    return mysql.connector.connect(
        host="82.180.143.66",
        user="u263681140_ADCET",
        password="Attendance@2026",
        database="u263681140_ADCET"
    )

employee_map = get_employee_map()
last_db_check = {}
detected_persons_start_time = {}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("\n📷 Starting Camera... Press 'q' on the camera window to exit.\n")

frame_count = 0
last_detected_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
        
    frame_count += 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process AI every 5 frames to keep the camera feed smooth without lagging
    if frame_count % 5 == 0:
        try:
            with HiddenPrints():
                faces = detector.detect_faces(image)
        except:
            faces = []
            
        detected_ids = []
        new_boxes = []
        
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0: continue
            
            face_img = cv2.resize(face_img, (160, 160))
            with HiddenPrints():
                embedding = embedder.embeddings([face_img])
            preds = svm_model.predict(embedding)
            prob = svm_model.predict_proba(embedding)
            
            confidence = np.max(prob)
            name = encoder.inverse_transform(preds)[0]
            
            if confidence >= threshold:
                row = employee_map[employee_map["Name"]==name]
                if not row.empty:
                    emp_id = row.iloc[0]["Employee_ID"]
                    detected_ids.append(emp_id)
                    label = f"{name} ({confidence*100:.2f}%)"
                    color = (0, 255, 0)
                else:
                    label = "Match Not Found"
                    color = (0, 0, 255)
            else:
                label = "Match Not Found"
                color = (0, 0, 255)
                
            new_boxes.append((x, y, w, h, color, label))
            
        last_detected_boxes = new_boxes
        detected_ids = list(set(detected_ids))
        
        current_time = time.time()
        
        # Clean up persons no longer in the frame
        for emp_id in list(detected_persons_start_time.keys()):
            if emp_id not in detected_ids:
                del detected_persons_start_time[emp_id]
                
        # Discover continuous 5-second detections
        ids_to_process = []
        for emp_id in detected_ids:
            if emp_id not in detected_persons_start_time:
                detected_persons_start_time[emp_id] = current_time
            elif (current_time - detected_persons_start_time[emp_id]) >= 5:
                ids_to_process.append(emp_id)
                
        # We only pass ids_to_process further down
        detected_ids_for_db = ids_to_process
    else:
        detected_ids_for_db = []
        
    # Draw boxes
    for box in last_detected_boxes:
        x, y, w, h, color, label = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # Database Logic
    if detected_ids_for_db:
        current_time = time.time()
        today = datetime.date.today().strftime("%Y-%m-%d")
        now = datetime.datetime.now()
        
        for emp_id in detected_ids_for_db:
            # 10s cooldown to prevent database spamming
            if emp_id in last_db_check and (current_time - last_db_check[emp_id]) < 10:
                continue
            last_db_check[emp_id] = current_time
            
            row = employee_map[employee_map["Employee_ID"] == emp_id]
            if row.empty: continue
            name = row.iloc[0]["Name"]
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT IN_Time, OUT_Time FROM attendance
                    WHERE Employee_ID=%s AND Date=%s
                """, (str(emp_id), today))
                record = cursor.fetchone()
                
                if record is None:
                    cursor.execute("""
                        INSERT INTO attendance 
                        (Employee_ID, Name, Date, IN_Time, OUT_Time, Hours)
                        VALUES (%s,%s,%s,%s,%s,%s)
                    """, (str(emp_id), str(name), today, now.strftime("%H:%M:%S"), None, 0))
                    print(f"✅ [{now.strftime('%H:%M:%S')}] {name} IN marked successfully!")
                else:
                    in_time, out_time = record
                    
                    in_time_str = str(in_time)
                    if len(in_time_str.split(':')) == 3:
                        in_dt = datetime.datetime.strptime(in_time_str[-8:].strip(), "%H:%M:%S") if len(in_time_str) >= 8 else datetime.datetime.strptime(in_time_str, "%H:%M:%S")
                    else:
                        in_dt = datetime.datetime.strptime(in_time_str, "%H:%M")

                    out_dt = datetime.datetime.strptime(now.strftime("%H:%M:%S"), "%H:%M:%S")
                    worked_seconds = (out_dt - in_dt).total_seconds()
                    if worked_seconds < 0: worked_seconds += 24*3600
                    worked_hours = worked_seconds / 3600
                    
                    if worked_hours >= 4:
                        cursor.execute("""
                            UPDATE attendance
                            SET OUT_Time=%s, Hours=%s
                            WHERE Employee_ID=%s AND Date=%s
                        """, (now.strftime("%H:%M:%S"), round(worked_hours,2), str(emp_id), today))
                        
                        if out_time is None:
                            print(f"🛑 [{now.strftime('%H:%M:%S')}] {name} OUT marked successfully! Total Hours: {round(worked_hours, 2)}")
                        else:
                            print(f"🔄 [{now.strftime('%H:%M:%S')}] {name} OUT time updated to latest scan! Total Hours: {round(worked_hours, 2)}")
                    else:
                        # Uncomment if you want console warnings for <4 hours
                        # print(f"⚠️ [{now.strftime('%H:%M:%S')}] {name} scanned again. Minimum 4 hours required for OUT.")
                        pass
                conn.commit()
            except Exception as e:
                print(f"⚠️ Database error for {name}: {e}")
            finally:
                if 'conn' in locals() and conn.is_connected():
                    conn.close()

    # Display the resulting frame
    cv2.imshow('WorkTrack Auto Attendance (Standalone)', frame)
    
    # Wait for 'q' to be pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nClosing Application...")
        break

cap.release()
cv2.destroyAllWindows()
