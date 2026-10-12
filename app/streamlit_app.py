"""Streamlit demo: ``streamlit run app/streamlit_app.py``."""

import os

import streamlit as st
from PIL import Image, UnidentifiedImageError

from scenetext.pipeline import OCRPipeline, draw_regions
from scenetext.utils import load_yaml

st.set_page_config(page_title="Scene Text Recognition", layout="wide")
st.title("Scene Text Recognition: YOLO26 + CRNN")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


@st.cache_resource
def get_pipeline() -> OCRPipeline:
    return OCRPipeline.from_config(load_yaml(os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml")))


pipe = get_pipeline()
pipe.det_conf = st.sidebar.slider("Detection confidence", 0.05, 0.95, pipe.det_conf, 0.05)
min_rec = st.sidebar.slider("Min. recognition confidence", 0.0, 1.0, 0.0, 0.05)

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], max_upload_size=10)
if upload:
    if upload.size > MAX_UPLOAD_BYTES:
        st.error("Image file is too large (maximum 10 MB).")
        st.stop()
    try:
        with Image.open(upload) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                st.error("Image dimensions are too large.")
                st.stop()
            img = source.convert("RGB")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()
    regions = pipe.predict(img, min_rec_conf=min_rec)
    left, right = st.columns(2)
    left.image(img, caption="Input", use_container_width=True)
    right.image(
        draw_regions(img, regions), caption=f"{len(regions)} regions", use_container_width=True
    )
    st.subheader("Recognised text")
    st.write(" ".join(r.text for r in regions) or "(nothing detected)")
    st.dataframe([r.to_dict() for r in regions], use_container_width=True)
