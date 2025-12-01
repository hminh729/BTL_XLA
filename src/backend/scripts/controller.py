import io
import zipfile
import uuid
import cv2

from typing import List
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from utils.imgUtils import *

async def post_imgs(
        files: List[UploadFile] = File(...),
        mode: str = Form("pencil")
    ):
    # Tạo buffer trong RAM để chứa file zip (không ghi ra ổ đĩa)
    zip_buffer = io.BytesIO()

    # Tạo file ZIP và nén từng ảnh vào
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:

            # Đọc file ảnh
            img_bytes = await file.read()

            # Convert từ bytes → numpy array
            img_array = np.frombuffer(img_bytes, np.uint8)

            # Giải mã ảnh OpenCV
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Resize ảnh xuống 40% để xử lý nhanh hơn
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))

            # Convert thành ảnh xám
            gray = rgb2gray_manual(img)

            # =============================
            #       CHỌN KIỂU EDGE
            # =============================
            if mode == "pencil":
                # Tạo ảnh âm bản (dùng tạo hiệu ứng bút chì)
                invert = 255 - gray

                # Làm mờ bằng Gaussian
                blur = gaussian_blur(gray, kernel_size=21, sigma=5)
                # blur = cv2.GaussianBlur(invert, (21, 21), sigmaX=0, sigmaY=0)

                # Tạo sketch bằng phép chia Dodge (gray / blur)
                sketch = cv2.divide(gray, 255 - blur, scale=256)

            elif mode == "sobel":
                # Làm mịn bằng Bilateral filter (giữ cạnh tốt)
                smooth = bilateral_filter_manual(gray, 5, 75, 75)

                # Lấy cạnh bằng Sobel
                edges = sobel_edge_manual(smooth, 50, 150)

                # Invert cho giống hiệu ứng vẽ tay
                sketch = 255 - edges

            elif mode == "prewitt":
                # Edge detection bằng Prewitt (hàm tự viết)
                edges = prewitt_edge_manual(gray, 50)
                sketch = 255 - edges

            elif mode == "laplacian":
                # Edge detection Laplacian
                edges = laplacian_edge_manual(gray, 30)
                sketch = 255 - edges

            else:
                sketch = gray  # fallback nếu mode sai

            # Encode ảnh sketch thành JPG để ghi vào zip
            ok, buf = cv2.imencode(".jpg", sketch)
            if ok:
                # Tạo filename unique
                filename = f"{mode}_{uuid.uuid4().hex[:8]}.jpg"

                # Ghi ảnh vào zip
                zipf.writestr(filename, buf.tobytes())

    # Reset pointer buffer về đầu
    zip_buffer.seek(0)

    # Trả file ZIP về cho client
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=sketch_results.zip"}
    )

async def preview_img(
    file: UploadFile = File(...),
    mode: str = Form("pencil")
):
    # Đọc ảnh upload
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content="Invalid image", status_code=400)

    # Resize để preview nhanh
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))

    # Convert sang grayscale
    gray = rgb2gray_manual(img)

    # =============================
    #       ÁP DỤNG MODE
    # =============================
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
        smooth = bilateral_filter_manual(gray, 5, 75, 75)
        edges = prewitt_edge_manual(smooth, 50)
        sketch = 255 - edges

    elif mode == "laplacian":
        smooth = bilateral_filter_manual(gray, 5, 75, 75)
        edges = laplacian_edge_manual(smooth, 30)
        sketch = 255 - edges

    else:
        sketch = gray

    # Encode ảnh ra JPG để trả về
    ok, buf = cv2.imencode(".jpg", sketch)
    return Response(content=buf.tobytes(), media_type="image/jpeg")