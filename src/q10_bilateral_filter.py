import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
f = cv.imread('Assignment/a1images/einstein.png', cv.IMREAD_GRAYSCALE)

sigma_s = 10    # spatial standard deviation
sigma_r = 25    # range (intensity) standard deviation
d = 9           # kernel diameter

# ── (a) Manual bilateral filter ────────────────────────────────────────────
def bilateral_filter(img, d, sigma_s, sigma_r):
    H, W = img.shape
    pad = d // 2
    img_pad = np.pad(img, pad, mode='reflect').astype(np.float64)
    output = np.zeros_like(img, dtype=np.float64)

    # Precompute spatial Gaussian weights for the kernel window
    ky, kx = np.meshgrid(np.arange(d) - pad, np.arange(d) - pad, indexing='ij')
    spatial_weights = np.exp(-(kx**2 + ky**2) / (2 * sigma_s**2))

    for i in range(H):
        for j in range(W):
            # Extract local patch
            patch = img_pad[i:i+d, j:j+d]
            # Range Gaussian: penalise large intensity differences
            intensity_diff = patch - img_pad[i+pad, j+pad]
            range_weights = np.exp(-(intensity_diff**2) / (2 * sigma_r**2))
            # Combined weight
            weights = spatial_weights * range_weights
            output[i, j] = np.sum(weights * patch) / np.sum(weights)

    return np.clip(output, 0, 255).astype(np.uint8)

# ── (b) Gaussian smoothing (OpenCV) ───────────────────────────────────────
g_gaussian = cv.GaussianBlur(f, (d, d), sigma_s)

# ── (c) Bilateral filter (OpenCV) ─────────────────────────────────────────
g_bilateral_cv = cv.bilateralFilter(f, d, sigma_r, sigma_s)

# ── (d) Manual bilateral filter ───────────────────────────────────────────
print('Running manual bilateral filter (this may take a moment)...')
g_bilateral_manual = bilateral_filter(f, d, sigma_s, sigma_r)
print('Done.')

# ── Display all results ────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(f'Bilateral Filter Comparison  (d={d}, σs={sigma_s}, σr={sigma_r})', fontsize=13)

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(g_gaussian, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('(b) Gaussian Blur\ncv.GaussianBlur')

ax[2].imshow(g_bilateral_cv, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('(c) Bilateral Filter\ncv.bilateralFilter')

ax[3].imshow(g_bilateral_manual, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('(d) Manual Bilateral Filter')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q10abcd_bilateral.png', dpi=150, bbox_inches='tight')
plt.show()
