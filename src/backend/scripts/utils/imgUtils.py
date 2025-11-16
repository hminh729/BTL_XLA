import numpy as np
import cv2

# =============================
#   CÁC HÀM XỬ LÝ ẢNH MANUAL
# =============================

# chuyen anh mau sang anh xam
def rgb2gray_manual(img):
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    return gray.astype(np.uint8)


def bilateral_filter_manual(img, ksize=5, sigma_color=75, sigma_space=75):
    # create padded image
    pad = ksize // 2
    img_pad = np.pad(img, pad, mode='edge')

    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    # ax = [-2, -1, 0, 1, 2]
    ax = np.linspace(-pad, pad, ksize)
    # tao luoi toan hoc cho khoang cach khong gian
    xx, yy = np.meshgrid(ax, ax)
    # Đây là thành phần Gaussian bộ lọc khoảng cách
    space_kernel = np.exp(-(xx**2 + yy**2)/(2*sigma_space**2))
    # lap  qua tung diem anh
    for i in range(h):
        for j in range(w):
            # lấy vùng hình ảnh có kích thước bằng kernel
            region = img_pad[i:i+ksize, j:j+ksize]
            # tính hiệu số giữa vùng hình ảnh và điểm trung tâm
            diff = region - img_pad[i+pad, j+pad]
            # Áp Gaussian màu:
            color_kernel = np.exp(-(diff**2)/(2*sigma_color**2))
            # Kết hợp cả hai thành phần
            combined = space_kernel * color_kernel
            # Chuẩn hóa và tính giá trị mới
            result[i,j] = np.sum(region * combined) / np.sum(combined)
    return result.astype(np.uint8)


def sobel_edge_manual(img, threshold1=50, threshold2=150):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # them padding cho anh
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

def prewitt_edge_manual(img, threshold=50):
    Kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    Ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

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

    G[G < threshold] = 0
    G[G >= threshold] = 255
    return G.astype(np.uint8)

def laplacian_edge_manual(img, threshold=30):
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
            G[i,j] = abs(np.sum(region * K))

    G[G < threshold] = 0
    G[G >= threshold] = 255
    return G.astype(np.uint8)

# def gaussian_kernel(ksize=21, sigma=5):
#     """Tạo kernel Gaussian 2D."""
#     ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
#     kernel = kernel / np.sum(kernel)
#     return kernel
#
#
# def gaussian_blur_manual(img, ksize=21, sigma=5):
#     """Blur ảnh xám với kernel Gaussian thủ công."""
#     kernel = gaussian_kernel(ksize, sigma)
#     pad = ksize // 2
#     img_pad = np.pad(img, pad, mode='edge')
#
#     h, w = img.shape
#     out = np.zeros_like(img, dtype=np.float32)
#
#     for i in range(h):
#         for j in range(w):
#             region = img_pad[i:i + ksize, j:j + ksize]
#             out[i, j] = np.sum(region * kernel)
#
#     return out.astype(np.uint8)
#
