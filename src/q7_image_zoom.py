import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -- Zoom function---------------------------------------------------------
def zoom_image(img, s, method='nearest'):
    H, W = img.shape
    new_H = int(H * s)
    new_W = int(W * s)
    output = np.zeros((new_H, new_W), dtype=np.uint8)

    for i in range(new_H):
        for j in range(new_W):
            # Map output pixel back to source coordinates
            src_y = i / s
            src_x = j / s

            if method == 'nearest':
                # (a) Nearest-neighbor: round to closest pixel
                sy = min(int(np.round(src_y)), H - 1)
                sx = min(int(np.round(src_x)), W - 1)
                output[i, j] = img[sy, sx]

            elif method == 'bilinear':
                # (b) Bilinear: weighted average of 4 surrounding pixels
                y0 = int(np.floor(src_y))
                x0 = int(np.floor(src_x))
                y1 = min(y0 + 1, H - 1)
                x1 = min(x0 + 1, W - 1)
                y0 = min(y0, H - 1)
                x0 = min(x0, W - 1)

                dy = src_y - np.floor(src_y)
                dx = src_x - np.floor(src_x)

                # Interpolate along x then y
                top    = (1 - dx) * img[y0, x0] + dx * img[y0, x1]
                bottom = (1 - dx) * img[y1, x0] + dx * img[y1, x1]
                output[i, j] = np.clip((1 - dy) * top + dy * bottom, 0, 255)

    return output


# -- Normalized SSD--------------------------------------------------------
def normalized_ssd(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    ssd = np.sum((img1 - img2) ** 2)
    return ssd / img1.size


# -- Test images: im01/02/03 pairs + taylor small/very_small vs original---
pairs = [
    ('Assignment/a1images/a1q7images/im01small.png',
     'Assignment/a1images/a1q7images/im01.png'),
    ('Assignment/a1images/a1q7images/im02small.png',
     'Assignment/a1images/a1q7images/im02.png'),
    ('Assignment/a1images/a1q7images/im03small.png',
     'Assignment/a1images/a1q7images/im03.png'),
    ('Assignment/a1images/a1q7images/taylor_small.jpg',
     'Assignment/a1images/a1q7images/taylor.jpg'),
    ('Assignment/a1images/a1q7images/taylor_very_small.jpg',
     'Assignment/a1images/a1q7images/taylor.jpg'),
]

for small_path, large_path in pairs:
    name = small_path.split('/')[-1].replace('.png', '').replace('.jpg', '')

    small = cv.imread(small_path, cv.IMREAD_GRAYSCALE)
    large = cv.imread(large_path, cv.IMREAD_GRAYSCALE)

    H_large, W_large = large.shape
    H_small, W_small = small.shape
    s = H_large / H_small   # zoom factor

    print(f'\n{name}: small={small.shape}, large={large.shape}, zoom factor s={s:.2f}')

    # (a) Nearest-neighbor zoom
    zoomed_nn = zoom_image(small, s, method='nearest')

    # (b) Bilinear zoom
    zoomed_bl = zoom_image(small, s, method='bilinear')

    # Resize to exactly match large image dimensions (in case of rounding)
    zoomed_nn = cv.resize(zoomed_nn, (W_large, H_large), interpolation=cv.INTER_NEAREST)
    zoomed_bl = cv.resize(zoomed_bl, (W_large, H_large), interpolation=cv.INTER_NEAREST)

    ssd_nn = normalized_ssd(zoomed_nn, large)
    ssd_bl = normalized_ssd(zoomed_bl, large)
    print(f'  Normalized SSD - Nearest-neighbor: {ssd_nn:.2f}')
    print(f'  Normalized SSD - Bilinear:         {ssd_bl:.2f}')

    # Display
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'{name}: Zoom factor s={s:.1f}', fontsize=13)

    ax[0].imshow(large, cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Original (large)')

    ax[1].imshow(small, cmap='gray', vmin=0, vmax=255)
    ax[1].set_title('Input (small)')

    ax[2].imshow(zoomed_nn, cmap='gray', vmin=0, vmax=255)
    ax[2].set_title(f'(a) Nearest-neighbor\nSSD={ssd_nn:.1f}')

    ax[3].imshow(zoomed_bl, cmap='gray', vmin=0, vmax=255)
    ax[3].set_title(f'(b) Bilinear\nSSD={ssd_bl:.1f}')

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(f'output/q7ab_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
