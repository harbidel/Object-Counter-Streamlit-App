"""
Object Counter — Streamlit App

Detects and counts any object(s) a user names in an uploaded image, using
YOLO-World (open-vocabulary object detection). No fixed class list — type
what you want counted at runtime.

Usage (local):
    pip install -r requirements-streamlit.txt
    streamlit run streamlit_app.py

Deploy: push this repo to GitHub, then create an app at
https://share.streamlit.io pointing at streamlit_app.py — see README.md
for the full walkthrough.
"""

from collections import Counter

import numpy as np
import streamlit as st
import supervision as sv
import torch
from PIL import Image
from ultralytics import YOLOWorld

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Object Counter",
    page_icon="🔎",
    layout="wide",
)

EXAMPLE_PROMPTS = [
    "person, car",
    "car, truck, bus, bicycle",
    "steel rod, stick",
    "bottle, cup, chair",
]

MODEL_WEIGHTS = "yolov8s-worldv2.pt"


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session/server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model…")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(MODEL_WEIGHTS)
    model.to(device)
    return model, device


model, DEVICE = load_model()
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)


# ---------------------------------------------------------------------------
# Core detection + counting logic
# ---------------------------------------------------------------------------
def count_objects(image_np, class_text, confidence):
    """
    Run open-vocabulary detection on `image_np` (RGB numpy array) for the
    classes in `class_text` (comma-separated). Returns the annotated image,
    a list of (class, count) rows, and the total count.
    """
    classes = [c.strip() for c in class_text.split(",") if c.strip()]

    # Workaround for a known ultralytics/CLIP device-mismatch bug: set_classes()
    # loads a fresh CLIP text encoder that can end up on a different device than
    # the rest of the model when running on GPU. Moving to CPU for this call,
    # then back to the target device, avoids the crash.
    model.to("cpu")
    model.set_classes(classes)
    model.to(DEVICE)

    results = model.predict(image_np, conf=confidence, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        table = [(cls, 0) for cls in classes]
        return image_np, table, 0

    detections = sv.Detections.from_ultralytics(result)

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = [result.names[i] for i in class_ids]
    counts = Counter(class_names)

    labels = [
        f"{result.names[cid]} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    table = [(cls, counts.get(cls, 0)) for cls in classes]
    total = sum(counts.values())

    return annotated, table, total


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { text-align: center; margin-bottom: 0; }
    .subtitle { text-align: center; color: var(--text-color-secondary, #888); margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .total-box {
        text-align: center;
        padding: 0.75rem 0;
        border-radius: 12px;
        background: rgba(127, 127, 127, 0.08);
        margin-bottom: 1rem;
    }
    .total-box .label { font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; opacity: 0.7; }
    .total-box .value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; }
    div.stButton > button[kind="primary"] { width: 100%; font-size: 1.05rem; padding: 0.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>🔎 Object Counter</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Upload a photo, tell it what to look for, and get a count — "
    "for any object, not just a fixed list.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "class_text" not in st.session_state:
    st.session_state.class_text = ""

left_col, right_col = st.columns(2, gap="large")

# -------------------- Left: inputs --------------------
with left_col:
    uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])

    st.text_input(
        "What should I count?",
        placeholder="e.g. car, person, steel rod",
        help="Separate multiple objects with commas.",
        key="class_text",
    )

    st.caption("Quick examples")
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, example in zip(example_cols, EXAMPLE_PROMPTS):
        if col.button(example, use_container_width=True):
            st.session_state.class_text = example
            st.rerun()

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.9,
        value=0.15,
        step=0.01,
        help="Lower catches more objects but risks false positives; higher is stricter.",
    )

    run_clicked = st.button("Count objects", type="primary", use_container_width=True)

    with st.expander("💡 Tips for better results"):
        st.markdown(
            "- Be descriptive: **\"steel rod\"** works better than **\"metal\"**.\n"
            "- Missing objects? Lower the confidence threshold.\n"
            "- Too many false positives? Raise the threshold or narrow the wording.\n"
            "- Objects that are tightly packed or overlapping (e.g. a bundle of rods) "
            "may be undercounted — this is a limitation of box-based detection."
        )

# -------------------- Right: outputs --------------------
with right_col:
    image_placeholder = st.empty()
    total_placeholder = st.empty()
    table_placeholder = st.empty()
    status_placeholder = st.empty()

    if uploaded_file is not None:
        preview = Image.open(uploaded_file).convert("RGB")
        image_placeholder.image(preview, caption="Preview", use_container_width=True)
    total_placeholder.markdown(
        "<div class='total-box'><div class='label'>Total objects</div>"
        "<div class='value'>—</div></div>",
        unsafe_allow_html=True,
    )

    if run_clicked:
        if uploaded_file is None:
            status_placeholder.warning("⚠️ Please upload an image first.")
        elif not st.session_state.class_text.strip():
            status_placeholder.warning("⚠️ Type at least one object to count, e.g. 'car, person'.")
        else:
            with st.spinner("Detecting…"):
                image_np = np.array(Image.open(uploaded_file).convert("RGB"))
                annotated, table, total = count_objects(
                    image_np, st.session_state.class_text, confidence
                )

            image_placeholder.image(annotated, caption="Detected objects", use_container_width=True)
            total_placeholder.markdown(
                f"<div class='total-box'><div class='label'>Total objects</div>"
                f"<div class='value'>{total}</div></div>",
                unsafe_allow_html=True,
            )
            table_placeholder.dataframe(
                {"Object": [row[0] for row in table], "Count": [row[1] for row in table]},
                use_container_width=True,
                hide_index=True,
            )
            if total == 0:
                status_placeholder.info(
                    "No objects detected. Try lowering the confidence threshold "
                    "or rewording the class names."
                )
            else:
                status_placeholder.success(
                    f"✅ Done — found {total} object{'s' if total != 1 else ''}."
                )
