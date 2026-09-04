---
title: Object Counter
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
short_description: Count any object in an image by typing what to look for
---

# Object Counter

Count any object in an image — people, cars, sticks, steel rods, or anything else you can name — using open-vocabulary object detection. Type what you want counted at runtime; no fixed class list, no training required.

## How it works

This project uses **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)**, an open-vocabulary detector. Instead of being limited to a fixed set of categories (like standard YOLO's 80 COCO classes), you give it a comma-separated list of object names at runtime, and it detects and counts each one in your image:

```
car, person, steel rod, stick
```

The model draws a bounding box around every instance it finds, and the app tallies the boxes per class.

## Repo structure

| File | What it is |
|---|---|
| [`app.py`](./app.py) | Gradio app. Run locally or deploy to Hugging Face Spaces. |
| [`requirements.txt`](./requirements.txt) | Dependencies for `app.py` (Gradio version). |
| [`streamlit_app.py`](./streamlit_app.py) | Streamlit app — same detection logic, Streamlit UI. Run locally or deploy to Streamlit Community Cloud. |
| [`requirements-streamlit.txt`](./requirements-streamlit.txt) | Dependencies for `streamlit_app.py`. |
| [`object_counter_gradio_app.ipynb`](./object_counter_gradio_app.ipynb) | Gradio app packaged as a Colab notebook (free GPU, no local setup). |
| [`object_counter_yolo_world.ipynb`](./object_counter_yolo_world.ipynb) | Earlier, simpler version with a hardcoded class list — kept as a minimal reference example. |

## Getting started

### Option A — Colab (no local setup, free GPU)

1. Open [`object_counter_gradio_app.ipynb`](./object_counter_gradio_app.ipynb) in [Google Colab](https://colab.research.google.com/)
2. Enable a GPU runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all cells (Runtime → Run all)
4. A Gradio link/interface appears at the bottom — upload an image, type the object(s) you want counted (comma-separated for multiple), and hit Submit

### Option B — Run locally (Gradio)

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`). Works without a GPU, just slower per image.

### Option C — Run locally (Streamlit)

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Opens automatically in your browser (usually `http://localhost:8501`).

### Option D — Deploy to Hugging Face Spaces (Gradio)

The YAML block at the top of this README configures the Space. To deploy:

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces), choosing **Gradio** as the SDK
2. Push this repo's contents to the Space (`app.py`, `requirements.txt`, this `README.md`)
3. The Space builds automatically from the config block above

### Option E — Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (if it isn't already)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, set the main file path to `streamlit_app.py`
4. Under **Advanced settings**, set the requirements file to `requirements-streamlit.txt` (or rename it to `requirements.txt` if you'd rather not specify it manually — Streamlit Cloud looks for that filename by default)
5. Deploy — first build takes a few minutes while it installs dependencies and downloads model weights

Streamlit Community Cloud's free tier is CPU-only, so detection will be slower than on a GPU (a few seconds per image rather than near-instant), but it works fine for a demo.

## Usage

- Type a single object (`car`) or several (`car, person, bicycle, dog`) — no fixed list.
- Be as descriptive as helps: `"steel rod"` detects better than `"metal"`; `"delivery van"` is more specific than `"vehicle"`.
- Adjust the confidence slider if results are off — lower catches more objects (with more false positives), higher is stricter.

## Tuning results

- **Missing objects?** Lower the confidence threshold (try 0.05–0.1), or use more descriptive class names.
- **Too many false positives?** Raise the threshold, or make class names more specific.
- **Tightly packed or overlapping objects undercounted** (e.g. a bundle of steel rods)? This is a known limitation of box-based detectors — separating touching/overlapping instances into individual boxes is hard. A density-map counting approach (e.g. CountGD), built specifically for packed/repetitive objects, is a planned follow-on for this case.

## Known issue: CUDA/CPU device mismatch

On GPU runtimes, `model.set_classes()` can throw:

```
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0
```

This is a device-handling bug in how YOLO-World's CLIP text encoder gets loaded, not a config error on your end. Both `app.py` and the Gradio notebook already include the fix: the model is moved to CPU before `set_classes()` and back to the target device (GPU or CPU) afterward. If you still hit this after pulling the latest version here, restart the runtime/kernel and re-run from the model-loading cell.

## Roadmap

- [x] Static image counting (YOLO-World)
- [x] User-defined classes at runtime (no hardcoded list)
- [x] Gradio interface for interactive use
- [x] Standalone Python script version
- [ ] Live webcam / video counting
- [ ] Density-map counting mode for tightly packed objects

## Requirements

Installed automatically via `requirements.txt` / the notebook's install cell:
- `ultralytics`
- `supervision`
- `gradio`
- `opencv-python-headless`
- `pillow`

## License

MIT — set in the Spaces config block above. Add a `LICENSE` file with the full MIT text if you also want it to show up as the repo's license on GitHub. Change the `license:` field in that block if you'd prefer something else.
