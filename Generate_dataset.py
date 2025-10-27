import cv2
import os
from insightface.app import FaceAnalysis
import numpy as np
import pandas as pd
import time # Import the time library

# Initialize the FaceAnalysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Get user details and camera URL dynamically ---
usn = input("Enter the USN for the individual: ")
name = input("Enter the Name for the individual: ")
url = input("Enter the phone camera URL (e.g., http://192.168.1.5:8080/video): ")
person_name = usn  # Use USN for the folder name to ensure uniqueness

if not usn or not name or not url:
    print("USN, Name, and URL cannot be empty. Exiting.")
    exit()

print(f"\nStarting dataset generation for:")
print(f"  USN: {usn}")
print(f"  Name: {name}")
print(f"  Camera URL: {url}")
print(f"  Images will be saved in folder: '{person_name}'")

# Set up folder paths
base_folder = r"D:/faceRec"  # Make sure this is your correct base path
face_data_folder = os.path.join(base_folder, "face_dataset")
eye_data_folder = os.path.join(base_folder, "eye_dataset")

# Create the specific person's folders
person_face_path = os.path.join(face_data_folder, person_name)
person_eye_path = os.path.join(eye_data_folder, person_name)
os.makedirs(person_face_path, exist_ok=True)
os.makedirs(person_eye_path, exist_ok=True)

# --- Camera Setup ---
print("\nAttempting to connect to camera...")
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("❌ Error: Could not open video stream. Please check the URL and your connection.")
    exit()

# --- NEW: Notification and 10-second delay ---
print("\n✅ Camera successfully connected!")
print("Get ready! Capturing will start in 10 seconds...")
time.sleep(10) # Wait for 10 seconds
print("Starting capture now! Look at the camera.")
# --- End of new section ---

img_id = 0
max_images = 100

while img_id < max_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, check camera URL and connection.")
        break

    # Display the live feed
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Capturing image {img_id + 1}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Capturing Images...', display_frame)

    # Face detection
    faces = app.get(frame)
    if faces:
        face = faces[0]
        box = face.bbox.astype(int)

        # Crop and save face
        cropped_face = frame[box[1]:box[3], box[0]:box[2]]
        face_img_path = os.path.join(person_face_path, f"img{img_id}.jpg")
        cv2.imwrite(face_img_path, cropped_face)

        # Eye detection (optional, if landmarks are available)
        if face.kps is not None:
            # Simple eye region extraction based on landmarks
            left_eye = face.kps[0].astype(int)
            right_eye = face.kps[1].astype(int)
            # Create a bounding box around both eyes
            eye_y_start = min(left_eye[1], right_eye[1]) - 20
            eye_y_end = max(left_eye[1], right_eye[1]) + 20
            eye_x_start = min(left_eye[0], right_eye[0]) - 20
            eye_x_end = max(left_eye[0], right_eye[0]) + 20
            cropped_eye = frame[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
            if cropped_eye.size > 0:
                eye_img_path = os.path.join(person_eye_path, f"eye_img{img_id}.jpg")
                cv2.imwrite(eye_img_path, cropped_eye)

        print(f"Successfully captured image {img_id + 1}")
        img_id += 1
    else:
        print("No face detected, please position yourself in the frame.")

    # Press Enter to break or wait for 100 images
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nCaptured {img_id} images and saved them.")
print("Now generating and appending embeddings...")

# --- Embedding Generation ---
# This part runs after capturing images. It appends the new person's data.

# Load existing data or create new files if they don't exist
try:
    face_embeddings_list = list(np.load('embeddings.npy'))
    eye_embeddings_list = list(np.load('eye_embeddings.npy'))
    labels_df = pd.read_csv('labels.csv')
    labels_list = labels_df.to_dict('records')
except FileNotFoundError:
    face_embeddings_list = []
    eye_embeddings_list = []
    labels_list = []
    print("Existing data files not found. Creating new ones.")

# Process the newly captured images
for i in range(img_id):
    face_path = os.path.join(person_face_path, f"img{i}.jpg")
    if os.path.exists(face_path):
        img = cv2.imread(face_path)
        faces = app.get(img)
        if len(faces) > 0:
            face_embeddings_list.append(faces[0].embedding)
            labels_list.append({'USN': usn, 'Name': name})

# Save the updated embeddings and labels
np.save('embeddings.npy', np.array(face_embeddings_list))
if eye_embeddings_list: # Only save if not empty
    np.save('eye_embeddings.npy', np.array(eye_embeddings_list))
pd.DataFrame(labels_list).to_csv('labels.csv', index=False)

print("\n✅ Success! New person's data has been added to the database.")