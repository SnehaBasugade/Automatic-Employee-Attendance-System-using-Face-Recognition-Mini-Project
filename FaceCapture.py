import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

person_name = input("Enter person name: ")
dataset_path = "dataset/" + person_name

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
max_images = 100

print("Press 'c' to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        face_img = frame[y:y+h, x:x+w]

        face_img = cv2.resize(face_img, (160,160))

        # Draw bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Show face preview
        cv2.imshow("Face Preview", face_img)

    # Instructions for user
    cv2.putText(frame,"Look Straight / Left / Right / Up / Down",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2)

    cv2.putText(frame,f"Images Captured: {count}/{max_images}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2)

    cv2.imshow("Dataset Creator", frame)

    key = cv2.waitKey(1)

    if key == ord('c') and len(faces) > 0:
        img_path = dataset_path + "/" + str(count) + ".jpg"
        cv2.imwrite(img_path, face_img)
        print("Saved:", img_path)
        count += 1

    if count >= max_images:
        break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()