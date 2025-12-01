import numpy as np
import cv2

# =============================
#   CÁC HÀM XỬ LÝ ẢNH MANUAL
# =============================


# ============================================================
# 1. HÀM CHUYỂN ẢNH RGB SANG ẢNH XÁM (KHÔNG DÙNG OpenCV)
# ============================================================
def rgb2gray_manual(img):
    # Tách các kênh B, G, R
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

    # Công thức chuyển sang grayscale theo chuẩn ITU-R BT.601
    gray = 0.114 * B + 0.587 * G + 0.299 * R

    # Trả về dạng uint8
    return gray.astype(np.uint8)



# ============================================================
# 2. BILATERAL FILTER MANUAL — GIỮ CẠNH, LÀM MƯỢT VÙNG MÀU
# ============================================================
def bilateral_filter_manual(img, ksize=5, sigma_color=75, sigma_space=75):

    # tạo padding cho ảnh
    pad = ksize // 2
    img_pad = np.pad(img, pad, mode='edge')

    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)

    # Tạo trục không gian (ví dụ: ax = [-2, -1, 0, 1, 2] khi ksize=5)
    ax = np.linspace(-pad, pad, ksize)

    # meshgrid để tạo ma trận tọa độ cho kernel không gian
    xx, yy = np.meshgrid(ax, ax)

    # Gaussian theo khoảng cách không gian
    space_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))

    # Duyệt từng pixel ảnh
    for i in range(h):
        for j in range(w):

            # Lấy vùng lân cận có kích thước bằng kernel
            region = img_pad[i:i+ksize, j:j+ksize]

            # Hiệu màu giữa region và pixel trung tâm
            diff = region - img_pad[i+pad, j+pad]

            # Gaussian theo màu (color)
            color_kernel = np.exp(-(diff**2) / (2 * sigma_color**2))

            # Kernel cuối = space * color
            combined = space_kernel * color_kernel

            # Pixel mới = tổng(region * kernel) / tổng kernel
            result[i, j] = np.sum(region * combined) / np.sum(combined)

    return result.astype(np.uint8)



# ============================================================
# 3. SOBEL FILTER (TÌM CẠNH) IMPLEMENTATION THỦ CÔNG
# ============================================================
def sobel_edge_manual(img, threshold1=50, threshold2=150):

    # Kernel Sobel X và Y
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    pad = 1
    img_pad = np.pad(img, pad, mode='edge')

    h, w = img.shape
    G = np.zeros_like(img, dtype=np.float32)

    # Convolution Sobel
    for i in range(h):
        for j in range(w):
            region = img_pad[i:i+3, j:j+3]

            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)

            # Magnitude của gradient
            G[i,j] = np.sqrt(gx**2 + gy**2)

    # Áp threshold để phân loại cạnh
    G[G < threshold1] = 0
    G[G > threshold2] = 255
    G[(G >= threshold1) & (G <= threshold2)] = 128

    return G.astype(np.uint8)



# ============================================================
# 4. PREWITT EDGE DETECTOR (TÌM CẠNH)
# ============================================================
def prewitt_edge_manual(img, threshold=50):

    # Prewitt Kernel
    Kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    Ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

    pad = 1
    img_pad = np.pad(img, pad, mode='edge')

    h, w = img.shape
    G = np.zeros_like(img, dtype=np.float32)

    # Convolution Prewitt
    for i in range(h):
        for j in range(w):
            region = img_pad[i:i+3, j:j+3]

            gx = np.sum(region * Kx)
            gy = np.sum(region * Ky)

            G[i, j] = np.sqrt(gx**2 + gy**2)

    # Threshold nhị phân hóa cạnh
    G[G < threshold] = 0
    G[G >= threshold] = 255

    return G.astype(np.uint8)



# ============================================================
# 5. LAPLACIAN EDGE DETECTOR (TÌM CẠNH)
# ============================================================
def laplacian_edge_manual(img, threshold=30):

    # Kernel Laplacian
    K = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    pad = 1
    img_pad = np.pad(img, pad, mode='edge')

    h, w = img.shape
    G = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = img_pad[i:i+3, j:j+3]

            # Lấy giá trị tuyệt đối của Laplacian
            G[i,j] = abs(np.sum(region * K))

    # Áp threshold
    G[G < threshold] = 0
    G[G >= threshold] = 255

    return G.astype(np.uint8)



# ============================================================
# 6. GAUSSIAN BLUR MANUAL (KHÔNG DÙNG OPENCV)
# ============================================================

# Tạo Gaussian kernel 2D
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Chuẩn hóa tổng = 1

    return kernel


# Gaussian Blur bằng convolution thủ công
def gaussian_blur(img, kernel_size=21, sigma=5):

    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2

    img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')

    height, width = img.shape
    output = np.zeros_like(img, dtype=np.float32)

    # Convolution với Gaussian kernel
    for i in range(height):
        for j in range(width):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output.astype(np.uint8)
