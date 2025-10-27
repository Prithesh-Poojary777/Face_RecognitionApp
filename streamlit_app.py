import streamlit as st
import cv2
import os
import time
import numpy as np
# Load pre-trained embeddings and labels
embeddings = np.load('embeddings.npy')
labels_df = pd.read_csv('labels.csv')


# --- Add this new line ---
eye_embeddings = np.load('eye_embeddings.npy')

# Page config
st.set_page_config(page_title="Fast Face Collector")
st.title("Image Collection for the AI Model")

# Inputs
folder = st.text_input("Folder to save images:", "")
total_images = st.number_input("Images to capture", 1, 300, 10)
interval = st.number_input("Interval (seconds)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

# Session state
if "count" not in st.session_state:
    st.session_state.count = 0
if "capturing" not in st.session_state:
    st.session_state.capturing = False

# Start button
if st.button("Start Auto-Capture"):
    if folder:
        os.makedirs(folder, exist_ok=True)
        st.session_state.capturing = True
    else:
        st.error("Please provide a folder name.")

# Capture logic
if st.session_state.capturing and st.session_state.count < total_images:
    cap = cv2.VideoCapture(0)
    time.sleep(0.2)  # Let the camera warm up

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.capturing = False
    else:
        ret, frame = cap.read()
        cap.release()

        if ret:
            filename = os.path.join(folder, f"img{st.session_state.count+1}.jpg")
            cv2.imwrite(filename, cv2.resize(frame, (400, 400)))
            st.session_state.count += 1

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     caption=f"Captured Image {st.session_state.count}",
                     use_container_width=True)

            if st.session_state.count < total_images:
                time.sleep(interval)  # Fast interval like 0.1
                st.rerun()
            else:
                st.success(f"✅ Done! {total_images} images saved to `{folder}`")
                st.session_state.count = 0
                st.session_state.capturing = False
        else:
            st.error("⚠️ Failed to capture image.")
            st.session_state.capturing = False
