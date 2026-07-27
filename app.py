import os
import cv2
import numpy as np
import time
import tempfile
import streamlit as st
from core.swapper import FaceSwapper
from core.utils import VideoProcessor

st.set_page_config(page_title="Face Swap", layout="wide")
st.title("Face Swap Studio")

@st.cache_resource(show_spinner=False)
def load_models():
    return FaceSwapper()

swapper = load_models()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Image", "Video"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Face")
    source_file = st.file_uploader("Upload source image", type=["jpg", "jpeg", "png"], key="src")
    if source_file:
        st.image(source_file, width=400)

with col2:
    st.subheader("Target")
    if mode == "Image":
        target_file = st.file_uploader("Upload target image", type=["jpg", "jpeg", "png"], key="tgt_img")
        if target_file:
            st.image(target_file, width=400)
    else:
        target_file = st.file_uploader("Upload target video", type=["mp4", "avi", "mov"], key="tgt_vid")
        if target_file:
            st.video(target_file)

st.markdown("---")

if st.button("Swap Faces", type="primary", use_container_width=True):
    if not source_file or not target_file:
        st.error("Please upload both source and target files")
    else:
        try:
            os.makedirs("outputs", exist_ok=True)

            if mode == "Image":
                st.info("Processing image...")

                img_source = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_COLOR)
                img_target = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)

                source_face = swapper.get_face(img_source)
                target_face = swapper.get_face(img_target)

                if not source_face or not target_face:
                    st.error("Could not detect faces in one or both images")
                else:
                    result = swapper.swap_faces(img_source, img_target)

                    output_path = f"outputs/images/swapped_{int(time.time())}.png"
                    cv2.imwrite(output_path, result)

                    st.success("Done!")
                    st.image(result, channels="BGR", width=400)

                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download Image",
                            f.read(),
                            file_name=os.path.basename(output_path),
                            mime="image/png"
                        )
            else:
                st.info("Processing video... this may take a while")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_src:
                    tmp_src.write(source_file.getbuffer())
                    tmp_src_path = tmp_src.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_tgt:
                    tmp_tgt.write(target_file.getbuffer())
                    tmp_tgt_path = tmp_tgt.name

                try:
                    output_path = f"outputs/videos/swapped_{int(time.time())}.mp4"
                    processor = VideoProcessor(swapper)
                    processor.process(tmp_tgt_path, output_path, tmp_src_path)

                    st.success("Done!")
                    with open(output_path, "rb") as f:
                        # st.video(f.read())
                        st.download_button(
                            "Download Video",
                            f.read(),
                            file_name=os.path.basename(output_path),
                            mime="video/mp4"
                        )
                finally:
                    for path in [tmp_src_path, tmp_tgt_path]:
                        if os.path.exists(path):
                            os.remove(path)

        except Exception as e:
            st.error(f"Error: {str(e)}")