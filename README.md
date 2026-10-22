# Mushroom Detection with YOLOv8s

A Streamlit web app for running a trained YOLOv8s object detector on uploaded images.

## Classes
- `CoW`
- `chanterelle`

## 1. Add your trained model
Copy your Experiment 1 checkpoint into this folder and name it `models/best.pt`.

Expected source on your computer:
`C:\Users\User\OneDrive\Documents\assign_5\runs\experiment_1\weights\best.pt`

The actual model weights are not included in this starter ZIP.

## 2. Run locally
Use a compatible Python version (Python 3.10 or 3.11 is recommended for deployment).

```bash
python -m venv .venv
```

Activate on Windows:
```bat
.venv\Scripts\activate
```

Install and launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

## 3. Deploy online
1. Create a GitHub repository, e.g. `mushroom-detection-yolov8`.
2. Upload `app.py`, `requirements.txt`, `.gitignore`, this README, and the model checkpoint if its size and licence permit.
3. Open Streamlit Community Cloud: https://share.streamlit.io/
4. Connect GitHub, choose the repository, set the main file to `app.py`, and deploy.
5. If the model file is too large for a normal GitHub upload, use Git LFS or a model-hosting service and update the app to retrieve it.

Never commit passwords, access tokens, private data, or personal photos.

## Limitations and safety
The Roboflow test split contains only 5 images and 9 labelled instances, so reported metrics are uncertain. Personal-image annotations use a generic `mushroom` class and are not compatible with the trained `CoW` and `chanterelle` classes. Predictions can be wrong; do not use this app to identify edible or poisonous mushrooms.
