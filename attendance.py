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
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WorkTrack Auto Attendance")
        self.root.geometry("1100x600")
        self.root.configure(bg="#2b2b2b")
        
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Left Panel (Video)
        self.left_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, expand=True, fill=tk.BOTH)
        
        self.title_label = tk.Label(self.left_frame, text="Live Camera Feed", font=("Segoe UI", 16, "bold"), bg="#2b2b2b", fg="white")
        self.title_label.pack(pady=(0, 10))
        
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Right Panel (Logs)
        self.right_frame = tk.Frame(self.root, bg="#2b2b2b", width=400)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)
        
        self.log_label = tk.Label(self.right_frame, text="Activity Logs", font=("Segoe UI", 16, "bold"), bg="#2b2b2b", fg="white")
        self.log_label.pack(pady=(0, 10))
        
        self.log_text = tk.Text(self.right_frame, width=50, height=30, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10), state=tk.DISABLED)
        self.log_text.pack(expand=True, fill=tk.BOTH)
        
        self.manage_btn = tk.Button(self.right_frame, text="⚙️ Manage Employees", font=("Segoe UI", 12, "bold"), bg="#8b5cf6", fg="white", command=self.open_manage_window)
        self.manage_btn.pack(pady=(10, 0), fill=tk.X)
        
        # Load Models Logic
        self.log("====================================")
        self.log("Loading Face Recognition Models...")
        self.root.update()
        
        with HiddenPrints():
            self.detector = MTCNN()
            self.embedder = FaceNet()
            self.svm_model = joblib.load("models/svm_model.pkl")
            self.encoder = joblib.load("models/label_encoder.pkl")
            
        self.log("Models loaded successfully. ✅")
        self.log("====================================")
        
        self.threshold = 0.60
        self.mapping_csv = "models/employee_mapping.csv"
        self.employee_map = self.get_employee_map()
        self.last_db_check = {}
        self.detected_persons_start_time = {}
        self.app_mode = "attendance"
        
        # Video Capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            
        self.log("\n📷 Starting Camera...\n")
        self.frame_count = 0
        self.last_detected_boxes = []
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start update loop
        self.update_frame()
        
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
    def get_employee_map(self):
        if os.path.exists(self.mapping_csv):
            return pd.read_csv(self.mapping_csv)
        return pd.DataFrame(columns=["Employee_ID","Name"])
        
    def get_db_connection(self):
        return mysql.connector.connect(
            host="82.180.143.66",
            user="u263681140_ADCET",
            password="Attendance@2026",
            database="u263681140_ADCET"
        )
        
    def update_status_db(self, state):
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE UpdateStatus SET state=%s WHERE id=1", (int(state),))
            conn.commit()
        except Exception as e:
            pass
        finally:
            if 'conn' in locals() and conn.is_connected():
                conn.close()
        
    def process_ai_logic(self, image):
        try:
            with HiddenPrints():
                faces = self.detector.detect_faces(image)
        except:
            faces = []
            
        detected_ids = []
        new_boxes = []
        
        current_frame_state = 0
        if len(faces) > 0:
            current_frame_state = 2
        
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0: continue
            
            face_img = cv2.resize(face_img, (160, 160))
            with HiddenPrints():
                embedding = self.embedder.embeddings([face_img])
            preds = self.svm_model.predict(embedding)
            prob = self.svm_model.predict_proba(embedding)
            
            confidence = np.max(prob)
            name = self.encoder.inverse_transform(preds)[0]
            
            if confidence >= self.threshold:
                row = self.employee_map[self.employee_map["Name"]==name]
                if not row.empty:
                    emp_id = row.iloc[0]["Employee_ID"]
                    detected_ids.append(emp_id)
                    label = f"{name} ({confidence*100:.2f}%)"
                    color = (0, 255, 0) # Green
                    current_frame_state = 1
                else:
                    label = "Match Not Found"
                    color = (255, 0, 0) # Red (in RGB, 255,0,0 is red)
            else:
                label = "Match Not Found"
                color = (255, 0, 0)
                
            new_boxes.append((x, y, w, h, color, label))
            
        self.last_detected_boxes = new_boxes
        detected_ids = list(set(detected_ids))
        
        current_time = time.time()
            
        # Clean up persons no longer in the frame to reset their timer
        for emp_id in list(self.detected_persons_start_time.keys()):
            if emp_id not in detected_ids:
                del self.detected_persons_start_time[emp_id]
            
        if getattr(self, "last_status_state", -1) != current_frame_state:
            self.last_status_state = current_frame_state
            threading.Thread(target=self.update_status_db, args=(current_frame_state,), daemon=True).start()
                
        # Determine who has been detected continuously for at least 5 seconds
        ids_to_process = []
        for emp_id in detected_ids:
            if emp_id not in self.detected_persons_start_time:
                self.detected_persons_start_time[emp_id] = current_time
            elif (current_time - self.detected_persons_start_time[emp_id]) >= 5:
                ids_to_process.append(emp_id)
        
        # Database Logic
        if ids_to_process:
            today = datetime.date.today().strftime("%Y-%m-%d")
            now = datetime.datetime.now()
            
            for emp_id in ids_to_process:
                if emp_id in self.last_db_check and (current_time - self.last_db_check[emp_id]) < 10:
                    continue
                self.last_db_check[emp_id] = current_time
                
                row = self.employee_map[self.employee_map["Employee_ID"] == emp_id]
                if row.empty: continue
                name = row.iloc[0]["Name"]
                
                try:
                    conn = self.get_db_connection()
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
                        self.log(f"✅ [{now.strftime('%H:%M:%S')}] {name} IN marked successfully!")
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
                                self.log(f"🛑 [{now.strftime('%H:%M:%S')}] {name} OUT marked! Total Hours: {round(worked_hours, 2)}")
                            else:
                                self.log(f"🔄 [{now.strftime('%H:%M:%S')}] {name} OUT updated! New Total Hours: {round(worked_hours, 2)}")
                    conn.commit()
                except Exception as e:
                    self.log(f"⚠️ Database error for {name}: {e}")
                finally:
                    if 'conn' in locals() and conn.is_connected():
                        conn.close()

    def open_manage_window(self):
        self.manage_win = tk.Toplevel(self.root)
        self.manage_win.title("Manage Employees")
        self.manage_win.geometry("500x600")
        self.manage_win.configure(bg="#2b2b2b")
        
        tk.Label(self.manage_win, text="Registered Employees", font=("Segoe UI", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)
        
        self.emp_listbox = tk.Listbox(self.manage_win, font=("Segoe UI", 12))
        self.emp_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.load_employees_list()
        
        delete_btn = tk.Button(self.manage_win, text="Delete Selected Employee", bg="#ef4444", fg="white", font=("Segoe UI", 10, "bold"), command=self.delete_selected_employee)
        delete_btn.pack(pady=5)
        
        tk.Label(self.manage_win, text="Register New Employee", font=("Segoe UI", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)
        
        form_frame = tk.Frame(self.manage_win, bg="#2b2b2b")
        form_frame.pack(pady=5)
        
        tk.Label(form_frame, text="ID:", bg="#2b2b2b", fg="white").grid(row=0, column=0, padx=5, pady=5)
        self.new_emp_id = tk.Entry(form_frame)
        self.new_emp_id.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(form_frame, text="Name:", bg="#2b2b2b", fg="white").grid(row=1, column=0, padx=5, pady=5)
        self.new_emp_name = tk.Entry(form_frame)
        self.new_emp_name.grid(row=1, column=1, padx=5, pady=5)
        
        capture_btn = tk.Button(self.manage_win, text="Capture Photos (100)", bg="#3b82f6", fg="white", font=("Segoe UI", 10, "bold"), command=self.capture_photos)
        capture_btn.pack(pady=5)
        
        train_btn = tk.Button(self.manage_win, text="Train Model", bg="#8b5cf6", fg="white", font=("Segoe UI", 10, "bold"), command=self.train_model)
        train_btn.pack(pady=10)

    def load_employees_list(self):
        self.emp_listbox.delete(0, tk.END)
        emp_map = self.get_employee_map()
        if not emp_map.empty:
            emp_map = emp_map.drop_duplicates(subset=["Employee_ID"])
            for _, row in emp_map.iterrows():
                self.emp_listbox.insert(tk.END, f"{row['Employee_ID']} - {row['Name']}")

    def delete_selected_employee(self):
        selection = self.emp_listbox.curselection()
        if not selection:
            messagebox.showwarning("Error", "Please select an employee to delete.")
            return
            
        emp_str = self.emp_listbox.get(selection[0])
        emp_id = emp_str.split(" - ")[0]
        name = emp_str.split(" - ")[1]
        
        if not messagebox.askyesno("Confirm", f"Delete {name}?"):
            return
            
        import shutil
        dataset_path = os.path.join("dataset", name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
            
        person_csv = "Person_Info_New.csv"
        if os.path.exists(person_csv):
            df = pd.read_csv(person_csv)
            df = df[df['Employee_ID'] != emp_id]
            df.to_csv(person_csv, index=False)
            
        mapping_csv = "models/employee_mapping.csv"
        if os.path.exists(mapping_csv):
            m_df = pd.read_csv(mapping_csv)
            m_df = m_df[m_df['Employee_ID'] != emp_id]
            m_df.to_csv(mapping_csv, index=False)
            
        self.employee_map = self.get_employee_map()
        self.load_employees_list()
        messagebox.showinfo("Success", "Employee deleted successfully.")

    def capture_photos(self):
        emp_id = self.new_emp_id.get().strip()
        name = self.new_emp_name.get().strip()
        
        if not emp_id or not name:
            messagebox.showwarning("Error", "Please enter ID and Name.")
            return
            
        self.capture_emp_id = emp_id
        self.capture_emp_name = name
        self.capture_count = 0
        self.app_mode = "capture"
        
        dataset_path = os.path.join("dataset", name)
        os.makedirs(dataset_path, exist_ok=True)
        messagebox.showinfo("Capture Started", "Look at the camera. 100 photos will be captured.")

    def save_new_employee(self):
        emp_id = self.capture_emp_id
        name = self.capture_emp_name
        
        person_csv = "Person_Info_New.csv"
        if os.path.exists(person_csv):
            df = pd.read_csv(person_csv)
        else:
            df = pd.DataFrame(columns=["Employee_ID","Name","Image_path"])
            
        new_rows = [{"Employee_ID": emp_id, "Name": name, "Image_path": f"dataset/{name}/{i}.jpg"} for i in range(100)]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(person_csv, index=False)
        
        mapping_csv = "models/employee_mapping.csv"
        if os.path.exists(mapping_csv):
            m_df = pd.read_csv(mapping_csv)
        else:
            m_df = pd.DataFrame(columns=["Employee_ID","Name"])
            
        m_df = pd.concat([m_df, pd.DataFrame([{"Employee_ID": emp_id, "Name": name}])], ignore_index=True)
        m_df.drop_duplicates(subset=["Employee_ID"], inplace=True)
        m_df.to_csv(mapping_csv, index=False)
        
        self.employee_map = self.get_employee_map()
        if hasattr(self, 'manage_win') and self.manage_win.winfo_exists():
            self.load_employees_list()

    def train_model(self, auto=False):
        if not auto and not messagebox.askyesno("Confirm", "Retrain AI model? This may take a minute."): return
        def run_training():
            try:
                self.log("Training started...")
                import subprocess, sys
                subprocess.run([sys.executable, "generate_embeddings.py"], check=True)
                subprocess.run([sys.executable, "train_model.py"], check=True)
                with HiddenPrints():
                    self.svm_model = joblib.load("models/svm_model.pkl")
                    self.encoder = joblib.load("models/label_encoder.pkl")
                self.log("Model retrained and loaded successfully! ✅")
                messagebox.showinfo("Success", "Model retrained successfully!")
            except Exception as e:
                self.log(f"Training error: {e}")
                messagebox.showerror("Error", f"Training failed: {e}")
        threading.Thread(target=run_training, daemon=True).start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.app_mode == "attendance":
                if self.frame_count % 5 == 0:
                    self.process_ai_logic(image)
                    
                for box in self.last_detected_boxes:
                    x, y, w, h, color, label = box
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            elif self.app_mode == "capture":
                if self.frame_count % 5 == 0:
                    try:
                        with HiddenPrints():
                            faces = self.detector.detect_faces(image)
                    except:
                        faces = []
                        
                    if len(faces) > 0:
                        x, y, w, h = faces[0]['box']
                        x, y = max(0, x), max(0, y)
                        face_img = frame[y:y+h, x:x+w].copy()
                        if face_img.size > 0:
                            import os
                            face_img = cv2.resize(face_img, (160, 160))
                            img_path = os.path.join("dataset", self.capture_emp_name, f"{self.capture_count}.jpg")
                            cv2.imwrite(img_path, face_img)
                            self.capture_count += 1
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
                        
                cv2.putText(image, f"Capturing: {self.capture_count}/100", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if getattr(self, "capture_count", 0) >= 100:
                    self.app_mode = "attendance"
                    self.save_new_employee()
                    messagebox.showinfo("Done", "100 photos captured successfully! Auto-starting model training now.")
                    self.train_model(auto=True)
            
            img_pil = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(10, self.update_frame)
        
    def on_closing(self):
        self.log("Closing Application...")
        self.root.update()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
