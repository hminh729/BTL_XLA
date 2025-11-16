import io
import zipfile
import uuid
import cv2
import numpy as np

from typing import List
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse

from utils.imgUtils import *

async def post_imgs(
        files: List[UploadFile] = File(...),
        mode: str = Form("pencil")
    ):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:

            img_bytes = await file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))

            gray = rgb2gray_manual(img)

            # =============================
            #       EDGE SELECTION
            # =============================
            if mode == "pencil":
                #  tao anh am ban de tao hieu ung but chi
                invert = 255 - gray
                # Gaussian blur
                # blur = gaussian_blur_manual(invert, ksize=21, sigma=0)
                blur = cv2.GaussianBlur(invert, (21, 21), sigmaX=0, sigmaY=0)
                sketch = cv2.divide(gray, 255 - blur, scale=256)

            elif mode == "sobel":
                smooth = bilateral_filter_manual(gray, 5, 75, 75)
                edges = sobel_edge_manual(smooth, 50, 150)
                sketch = 255 - edges

            elif mode == "prewitt":
                edges = prewitt_edge_manual(gray, 50)
                sketch = 255 - edges



            elif mode == "laplacian":
                edges = laplacian_edge_manual(gray, 30)
                sketch = 255 - edges

            else:
                sketch = gray

            ok, buf = cv2.imencode(".jpg", sketch)
            if ok:
                filename = f"{mode}_{uuid.uuid4().hex[:8]}.jpg"
                zipf.writestr(filename, buf.tobytes())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=sketch_results.zip"}
    )


from fastapi.responses import Response

async def preview_img(
    file: UploadFile = File(...),
    mode: str = Form("pencil")
):
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content="Invalid image", status_code=400)

    # Resize tránh lag
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))
    gray = rgb2gray_manual(img)
    if mode == "pencil":
        invert = 255 - gray
        # blur = gaussian_blur_manual(invert, ksize=21, sigma=0)
        blur = cv2.GaussianBlur(invert, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

    elif mode == "sobel":
        smooth = bilateral_filter_manual(gray, 5, 75, 75)
        edges = sobel_edge_manual(smooth, 50, 150)
        sketch = 255 - edges

    elif mode == "prewitt":
        edges = prewitt_edge_manual(gray, 50)
        sketch = 255 - edges



    elif mode == "laplacian":
        edges = laplacian_edge_manual(gray, 30)
        sketch = 255 - edges

    else:
        sketch = gray  # fallback

    ok, buf = cv2.imencode(".jpg", sketch)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
