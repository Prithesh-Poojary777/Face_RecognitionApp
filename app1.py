import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# --- GLOBAL INITIALIZATION (runs only once) ---
@st.cache_resource
def load_models_and_data():
    try:
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        embeddings = np.load('embeddings.npy')
        labels_df = pd.read_csv('labels.csv')
        return face_app, embeddings, labels_df
    except FileNotFoundError:
        st.error("Error: One or more data files (embeddings.npy, labels.csv) not found.")
        st.stop()

face_app, embeddings, labels_df = load_models_and_data()

# --- Shared state for results (for Laptop Camera WebRTC) ---
lock = threading.Lock()
result_container = {"notification": "Stream not running.", "name": None, "usn": None}

# --- VideoTransformer Class (for Laptop Camera WebRTC) ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self, face_app, embeddings, labels_df):
        self.face_app = face_app
        self.embeddings = embeddings
        self.labels_df = labels_df

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        faces = self.face_app.get(img)
        notification, name, usn = "Searching for face...", None, None

        if faces:
            face = faces[0]
            new_embedding = face.embedding
            similarities = cosine_similarity(new_embedding.reshape(1, -1), self.embeddings).flatten()
            max_sim_index = np.argmax(similarities)
            
            box = face.bbox.astype(int)
            if similarities[max_sim_index] > 0.5:
                matched_label = self.labels_df.iloc[max_sim_index]
                name, usn = matched_label['Name'], matched_label['USN']
                notification = "✅ Person Identified"
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                notification = "⚠️ Person Not Identified"
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(img, "Not Identified", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        with lock:
            result_container.update({"notification": notification, "name": name, "usn": usn})
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Face Recognition Function (used by both Upload and Phone Camera) ---
def recognize_face(frame, face_app, embeddings, labels_df):
    faces = face_app.get(frame)
    if not faces:
        return frame, "Searching for face...", None, None

    face = faces[0]
    new_embedding = face.embedding
    similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings).flatten()
    max_sim_index = np.argmax(similarities)
    
    box = face.bbox.astype(int)
    if similarities[max_sim_index] > 0.5:
        matched_label = labels_df.iloc[max_sim_index]
        name, usn = matched_label['Name'], matched_label['USN']
        notification = "✅ Person Identified"
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        name, usn = None, None
        notification = "⚠️ Person Not Identified"
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(frame, "Not Identified", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, notification, name, usn

# --- Streamlit UI ---
st.set_page_config(page_title="Face Recognition", layout="wide")
st.title("👁️ Face Recognition System")

st.sidebar.title("Options")
recognition_method = st.sidebar.radio("Choose Recognition Method", ('Upload Image', 'Live Camera Stream'))

# --- UPLOAD IMAGE option ---
if recognition_method == 'Upload Image':
    st.markdown("### 🖼️ Upload an Image for Recognition")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        faces = face_app.get(img)
        
        if not faces:
            st.warning("No faces detected in the uploaded image.")
        else:
            st.info(f"Found {len(faces)} face(s). Identifying...")
            
            img_with_boxes = img.copy()
            identified_persons = []

            for face in faces:
                new_embedding = face.embedding
                similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings).flatten()
                max_sim_index = np.argmax(similarities)
                box = face.bbox.astype(int)

                if similarities[max_sim_index] > 0.5:
                    matched_label = labels_df.iloc[max_sim_index]
                    name, usn = matched_label['Name'], matched_label['USN']
                    identified_persons.append({'name': name, 'usn': usn})
                    cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(img_with_boxes, f"{name}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(img_with_boxes, "Not Identified", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            st.image(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
            
            if identified_persons:
                st.subheader("✅ Identified Individuals")
                unique_persons = pd.DataFrame(identified_persons).drop_duplicates().to_dict('records')
                for person in unique_persons:
                    st.write(f"**Name:** {person['name']}, **USN:** {person['usn']}")
            else:
                st.subheader("⚠️ No known individuals were identified.")

# --- LIVE CAMERA STREAM option ---
elif recognition_method == 'Live Camera Stream':
    st.markdown("### 📷 Live Camera Feed")
    camera_choice = st.radio("Select Camera Source", ('Laptop Camera', 'Phone Camera (URL)'), key="camera_choice")
    st.markdown("---")

    # --- Logic for Laptop Camera ---
    if camera_choice == 'Laptop Camera':
        webrtc_ctx = webrtc_streamer(
            key="laptop-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoTransformer(face_app, embeddings, labels_df),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        st.subheader("Recognition Status")
        if not webrtc_ctx.state.playing:
            st.info("Stream is not running. Press 'Start' on the video player above.")
        else:
            with lock:
                name, usn, notification = result_container.get("name"), result_container.get("usn"), result_container.get("notification")
            
            if name and usn:
                st.success(f"**Status:** {notification}")
                col1, col2 = st.columns(2)
                col1.metric("Name", name)
                col2.metric("USN", usn)
            else:
                st.warning(f"**Status:** {notification}")

    # --- Logic for Phone Camera ---
    elif camera_choice == 'Phone Camera (URL)':
        camera_url = st.text_input("Enter Phone Camera URL:", "http://19.168.1.5:8080/video")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Phone Stream"):
                if camera_url:
                    st.session_state.phone_stream_active = True
                else:
                    st.error("Please enter a valid camera URL.")
        with col2:
            if st.button("Stop Phone Stream"):
                st.session_state.phone_stream_active = False

        if 'phone_stream_active' in st.session_state and st.session_state.phone_stream_active:
            cap = None
            try:
                with st.spinner("Connecting to phone camera..."):
                    cap = cv2.VideoCapture(camera_url)
                
                if not cap.isOpened():
                    st.error("❌ Could not open video stream. Check URL and connection.")
                    st.session_state.phone_stream_active = False
                else:
                    st.success("✅ Connection successful! Streaming...")
                    image_placeholder = st.empty()
                    result_placeholder = st.empty()
                    
                    while 'phone_stream_active' in st.session_state and st.session_state.phone_stream_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream ended or failed to grab frame.")
                            st.session_state.phone_stream_active = False
                            break
                        
                        processed_frame, notification, name, usn = recognize_face(frame, face_app, embeddings, labels_df)
                        image_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                        
                        with result_placeholder.container():
                            if name and usn:
                                st.success(f"**Status:** {notification}")
                                col1, col2 = st.columns(2)
                                col1.metric("Name", name)
                                col2.metric("USN", usn)
                            else:
                                st.warning(f"**Status:** {notification}")
            finally:
                if cap:
                    cap.release()