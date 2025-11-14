import io
import zipfile
import uuid
import cv2
import numpy as np
from typing import List
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
OUTPUT_DIR = r"E:\PTIT\Ky1nam4\XuLyAnh\BTL\BTL\src\backend\images\output"


# =============================
#   CÁC HÀM XỬ LÝ ẢNH MANUAL
# =============================

def rgb2gray_manual(img):
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    return gray.astype(np.uint8)

def bilateral_filter_manual(img, ksize=5, sigma_color=75, sigma_space=75):
    pad = ksize // 2
    img_pad = np.pad(img, pad, mode='edge')
    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)

    ax = np.linspace(-pad, pad, ksize)
    xx, yy = np.meshgrid(ax, ax)
    space_kernel = np.exp(-(xx**2 + yy**2)/(2*sigma_space**2))

    for i in range(h):
        for j in range(w):
            region = img_pad[i:i+ksize, j:j+ksize]
            diff = region - img_pad[i+pad, j+pad]
            color_kernel = np.exp(-(diff**2)/(2*sigma_color**2))
            combined = space_kernel * color_kernel
            result[i,j] = np.sum(region * combined) / np.sum(combined)

    return result.astype(np.uint8)

def sobel_edge_manual(img, threshold1=50, threshold2=150):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    pad = 1
    img_pad = np.pad(img, pad, mode='edge')
    h, w = img.shape
    G = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = img_pad[i:i+3, j:j+3]
            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)
            G[i,j] = np.sqrt(gx**2 + gy**2)

    G[G < threshold1] = 0
    G[G > threshold2] = 255
    G[(G >= threshold1) & (G <= threshold2)] = 128

    return G.astype(np.uint8)


# =============================
#   API XỬ LÝ NHIỀU ẢNH
# =============================

async def post_imgs(files: List[UploadFile] = File(...)):
    # Tạo buffer chứa file ZIP
    zip_buffer = io.BytesIO()

    # Mở ZIP trong RAM
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:

        for file in files:
            # Đọc ảnh
            img_bytes = await file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Resize tránh lag
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*0.4), int(h*0.4)))

            # Xử lý SKETCH
            gray = rgb2gray_manual(img)
            smooth = bilateral_filter_manual(gray, 5, 75, 75)
            edges = sobel_edge_manual(smooth)
            sketch = 255 - edges

            # Encode JPEG
            ok, buffer = cv2.imencode(".jpg", sketch)
            if not ok:
                continue

            # Tạo tên file trong ZIP
            new_name = f"sketch_{uuid.uuid4().hex[:8]}.jpg"

            # Ghi file vào ZIP
            zipf.writestr(new_name, buffer.tobytes())

    # Dịch con trỏ về đầu buffer
    zip_buffer.seek(0)

    # Trả về file ZIP cho client
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=sketch_results.zip"
        }
    )
