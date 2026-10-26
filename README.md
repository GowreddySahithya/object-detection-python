# Object Detection with Python — Web Demo

This Streamlit app lets a user upload an image, runs a pre-trained YOLOv8n model, displays object labels and bounding boxes, and provides a download button for the annotated image.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Publish a live link

1. Sign in to GitHub and create a new repository, for example `object-detection-python`.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository, branch, and `app.py`.
5. Click **Deploy**. Wait for the build to finish; Streamlit will provide a public URL.

The first run may take a little longer because the pre-trained model weights need to download. Do not commit model weights to GitHub unless you specifically intend to host them there.

## Important

The app performs inference using a pre-trained model; it does not train a custom model. A public URL can only be generated after you deploy it through an account you control.
