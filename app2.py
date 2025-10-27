import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from fpdf import FPDF
import os
import uuid
from PIL import Image

# Load pre-trained embeddings and labels
embeddings = np.load('embeddings.npy')
labels_df = pd.read_csv('labels.csv')

# Initialize FaceAnalysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Streamlit App Config and Title
st.set_page_config(page_title="AI Detail Extraction", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Face Detection and Detail Extraction using AI</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Reset matches for each new upload
    matched_faces = []

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # BGR format

    st.markdown("### 📷 Uploaded Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    faces = app.get(img)

    if len(faces) == 0:
        st.error("❌ No face detected. Please try another image.")
    else:
        for i, face in enumerate(faces):
            st.markdown(f"<hr><h4 style='color:#2196F3;'>👤 Face {i+1} Result</h4>", unsafe_allow_html=True)

            embedding = face.normed_embedding.reshape(1, -1)
            similarities = cosine_similarity(embedding, embeddings)[0]
            max_sim = np.max(similarities)
            max_index = np.argmax(similarities)

            # Crop face safely (ensure coords inside image)
            x1, y1, x2, y2 = map(int, face.bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # handle potential degenerate crops
            if x2 <= x1 or y2 <= y1:
                face_img = img.copy()
            else:
                face_img = img[y1:y2, x1:x2]

            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(face_img_rgb, caption="Detected Face", use_container_width=True)

            with col2:
                if max_sim > 0.4:
                    full_info = labels_df.iloc[max_index]['USN']
                    parts = full_info.split("_")
                    usn_part = parts[0] if len(parts) > 0 else "Unknown"
                    name_part = parts[1] if len(parts) > 1 else "Unknown"
                    dept_part = parts[2] if len(parts) > 2 else "Unknown"

                    st.success(
                        f"✅ **Match Found**\n\n"
                        f"**USN:** {usn_part}\n\n"
                        f"**Name:** {name_part}\n\n"
                        f"**Department:** {dept_part}\n\n"
                    )

                    # Save face for potential future use, but PDF will contain only USNs as requested
                    matched_faces.append({
                        "usn": usn_part,
                        "name": name_part,
                        "dept": dept_part,
                        "similarity": max_sim,
                        "face_image": face_img_rgb
                    })
                else:
                    st.warning(
                        f"⚠️ **Relevant Data Not Available (Person may be outside your organization)**"
                    )

        # Display exact count of matched people
        total_matches = len(matched_faces)
        if total_matches > 0:
            st.info(f"🧾 **Total matched people:** {total_matches}")

            # Sort matches by USN before creating PDF
            matched_faces_sorted = sorted(matched_faces, key=lambda x: x['usn'])

            # Generate PDF containing only the USNs (one per line) and the total count on the first page
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # First page: title + total count
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="CSE(IOT) Attendance Report", ln=True)
            pdf.ln(4)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, txt=f"Total Attendance: {total_matches}", ln=True)
            pdf.ln(6)

            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, txt="Students Present:", ln=True)
            pdf.ln(4)

            # Write USNs, one per line
            for idx, match in enumerate(matched_faces_sorted, start=1):
                # Make sure text doesn't overflow the line width
                pdf.multi_cell(0, 8, txt=f"{idx}. {match['usn']}")

            # Save PDF to a temporary file and offer for download
            os.makedirs('/tmp', exist_ok=True)
            pdf_output_path = f"/tmp/match_report_{uuid.uuid4().hex}.pdf"
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as f:
                st.download_button("📄 Download Final Report (PDF) - USNs Only", f, file_name="match_report_usns.pdf", mime="application/pdf")
        else:
            st.info("No matches to include in the report.")
else:
    st.info("Upload an image to start face detection and matching.")
