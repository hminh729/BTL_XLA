import io
import zipfile
import uuid
import cv2
import numpy as np

from typing import List
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse

from utils.imgUtils import (
    rgb2gray_manual,
    bilateral_filter_manual,
    sobel_edge_manual
)


async def post_imgs(
        files: List[UploadFile] = File(...),
        mode: str = Form("pencil")  # 👈 Thêm mode
):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:

            img_bytes = await file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Resize tránh lag
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))

            # ============================================
            # MODE 1: PENCIL SKETCH (Gaussian Blur)
            # ============================================
            if mode == "pencil":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                invert = 255 - gray
                blur = cv2.GaussianBlur(invert, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blur, scale=256)

                ok, buf = cv2.imencode(".jpg", sketch)
                if ok:
                    filename = f"pencil_{uuid.uuid4().hex[:8]}.jpg"
                    zipf.writestr(filename, buf.tobytes())

            # ============================================
            # MODE 2: SOBEL SKETCH (manual)
            # ============================================
            else:
                gray = rgb2gray_manual(img)
                smooth = bilateral_filter_manual(gray, 5, 75, 75)
                edges = sobel_edge_manual(smooth, 50, 150)
                sketch = 255 - edges

                ok, buf = cv2.imencode(".jpg", sketch)
                if ok:
                    filename = f"sobel_{uuid.uuid4().hex[:8]}.jpg"
                    zipf.writestr(filename, buf.tobytes())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=sketch_results.zip"}
    )
