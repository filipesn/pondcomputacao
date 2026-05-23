import numpy as np
import cv2
import os

# ==========================================
# 1. FUNÇÕES BASE
# ==========================================

def gerar_kernel_gaussiano(size, sigma=None):
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def convolve2d(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    # Cálculo de padding assimétrico para suportar kernels pares e impares
    pad_h_antes = k_h // 2
    pad_h_depois = k_h - pad_h_antes - 1
    pad_w_antes = k_w // 2
    pad_w_depois = k_w - pad_w_antes - 1

    padded = np.pad(image, ((pad_h_antes, pad_h_depois), (pad_w_antes, pad_w_depois)), mode='edge')
    shape = (i_h, i_w, k_h, k_w)
    strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))

def average_pooling(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = h - (h % scale), w - (w % scale)
    img_c = img[:new_h, :new_w]
    if len(img_c.shape) == 3:
        return img_c.reshape(new_h // scale, scale, new_w // scale, scale, 3).mean(axis=(1, 3)).astype(np.uint8)
    return img_c.reshape(new_h // scale, scale, new_w // scale, scale).mean(axis=(1, 3)).astype(np.uint8)

def aplicar_clahe_numpy(gray_img, clip_limit=2.2, grid_size=(8, 8)):
    img = np.clip(gray_img, 0, 255).astype(np.uint8)
    h, w = img.shape
    gy, gx = grid_size

    pad_y = (gy - h % gy) % gy
    pad_x = (gx - w % gx) % gx
    img_pad = np.pad(img, ((0, pad_y), (0, pad_x)), mode='reflect')
    hp, wp = img_pad.shape
    ty, tx = hp // gy, wp // gx

    cdfs = np.zeros((gy, gx, 256), dtype=np.float32)
    clip_val = clip_limit * ty * tx / 256.0

    for i in range(gy):
        for j in range(gx):
            tile = img_pad[i*ty:(i+1)*ty, j*tx:(j+1)*tx]
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))

            excess = np.maximum(hist - clip_val, 0)
            hist = np.minimum(hist, clip_val)
            hist += excess.sum() / 256.0

            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255.0 / (cdf.max() - cdf.min() + 1e-7)
            cdfs[i, j] = cdf

    padded_cdfs = np.pad(cdfs, ((1, 1), (1, 1), (0, 0)), mode='edge')

    y_indices = np.arange(hp)
    x_indices = np.arange(wp)

    y_map = (y_indices - ty / 2.0) / ty
    x_map = (x_indices - tx / 2.0) / tx

    y1 = np.floor(y_map).astype(int)
    y2 = y1 + 1
    x1 = np.floor(x_map).astype(int)
    x2 = x1 + 1

    wy = (y_map - y1).reshape(-1, 1)
    wx = (x_map - x1).reshape(1, -1)

    y1 += 1; y2 += 1
    x1 += 1; x2 += 1

    y1_grid, x1_grid = np.meshgrid(y1, x1, indexing='ij')
    y2_grid, x2_grid = np.meshgrid(y2, x2, indexing='ij')

    val_tl = padded_cdfs[y1_grid, x1_grid, img_pad]
    val_tr = padded_cdfs[y1_grid, x2_grid, img_pad]
    val_bl = padded_cdfs[y2_grid, x1_grid, img_pad]
    val_br = padded_cdfs[y2_grid, x2_grid, img_pad]

    val_t = val_tl * (1 - wx) + val_tr * wx
    val_b = val_bl * (1 - wx) + val_br * wx
    result_pad = val_t * (1 - wy) + val_b * wy

    return result_pad[:h, :w]

def otimizar_iluminacao_local(gray_img, kernel_blur, alpha=1.5):
    img_clahe = aplicar_clahe_numpy(gray_img, clip_limit=2.3, grid_size=(8, 8))
    img_blur = convolve2d(img_clahe, kernel_blur)
    img_sharpened = img_clahe + alpha * (img_clahe - img_blur)
    return np.clip(img_sharpened, 0, 255)

def aplicar_gamma(gray_img, gamma):
    return 255.0 * ((gray_img / 255.0) ** gamma)

def non_maximum_suppression(magnitude, theta):
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    padded_mag = np.pad(magnitude, 1, mode='constant')

    for i in range(1, h-1):
        for j in range(1, w-1):
            q, r = 255, 255
            ang = angle[i, j]
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q, r = padded_mag[i+1, j+2], padded_mag[i+1, j]
            elif (22.5 <= ang < 67.5):
                q, r = padded_mag[i+2, j], padded_mag[i, j+2]
            elif (67.5 <= ang < 112.5):
                q, r = padded_mag[i+2, j+1], padded_mag[i, j+1]
            elif (112.5 <= ang < 157.5):
                q, r = padded_mag[i, j], padded_mag[i+2, j+2]

            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                nms[i,j] = magnitude[i,j]
    return nms

def histerese(img_nms, low_thresh, high_thresh):
    res_strong = (img_nms >= high_thresh).astype(np.uint8)
    res_weak = ((img_nms >= low_thresh) & (img_nms < high_thresh)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.float32)

    while True:
        dilated = convolve2d(res_strong.astype(np.float32), kernel)
        new_strong = np.logical_and(dilated > 0, res_weak)
        if not np.any(new_strong): break
        res_strong = np.logical_or(res_strong, new_strong)
        res_weak = np.logical_and(res_weak, np.logical_not(new_strong))
    return res_strong

# ==========================================
# 2. ALGORITMO DE RASTREAMENTO
# ==========================================
def extract_paths(edges):
    paths = []
    visited = np.zeros_like(edges, dtype=bool)
    h, w = edges.shape
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    for i in range(h):
        for j in range(w):
            if edges[i, j] > 0 and not visited[i, j]:
                path = []
                curr = (i, j)
                while curr:
                    ci, cj = curr
                    visited[ci, cj] = True
                    path.append((ci, cj))
                    next_pixel = None
                    for di, dj in dirs:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if edges[ni, nj] > 0 and not visited[ni, nj]:
                                next_pixel = (ni, nj)
                                break
                    curr = next_pixel

                if len(path) > 3:
                    paths.append(path)
    return paths

def map_to_turtlesim(paths, img_shape):
    h, w = img_shape
    TS_MAX = 11.0
    scale = 11.0 / max(h, w)
    offset_x = (TS_MAX - (w * scale)) / 2
    offset_y = (TS_MAX - (h * scale)) / 2

    ts_paths = []
    for path in paths:
        ts_path = []
        for (i, j) in path:
            x = offset_x + j * scale
            y = offset_y + (h - 1 - i) * scale
            ts_path.append((x, y))

        ts_paths.append(ts_path)
    return ts_paths

# ==========================================
# 3. PIPELINE PRINCIPAL
# ==========================================
def processar_imagem_para_turtlesim(caminho_da_imagem):
    img_bgr = cv2.imread(caminho_da_imagem)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho_da_imagem}")

    img_rgb = img_bgr[..., ::-1] 

    h, w = img_rgb.shape[:2]
    escala = max(1, max(h, w) // 1000)
    if escala > 1:
        img_rgb = average_pooling(img_rgb, escala)

    img_gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])

    gauss_5x5 = gerar_kernel_gaussiano(5)
    gauss_14x14 = gerar_kernel_gaussiano(14)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gray_otimizado = otimizar_iluminacao_local(img_gray, gauss_5x5, alpha=2.5)
    img_gamma = aplicar_gamma(gray_otimizado, 0.4)
    blurred = convolve2d(img_gamma, gauss_14x14)

    Gx = convolve2d(blurred, Kx)
    Gy = convolve2d(blurred, Ky)

    magnitude = (np.sqrt(Gx**2 + Gy**2) / (np.max(np.sqrt(Gx**2 + Gy**2)) + 1e-7)) * 255.0
    theta = np.arctan2(Gy, Gx)

    bordas_finas = non_maximum_suppression(magnitude, theta)

    bordas_finais = histerese(bordas_finas, 35, 95)

    caminhos = extract_paths(bordas_finais)
    caminhos_turtlesim = map_to_turtlesim(caminhos, bordas_finais.shape)

    return caminhos_turtlesim
