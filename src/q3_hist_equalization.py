import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the runway image
f = cv.imread('Assignment/runway.png', cv.IMREAD_GRAYSCALE)

# -- Custom histogram equalization function-------------------------------
# Formula (lecture slide 54): s_k = (L-1)/MN * sum(n_j, j=0..k)
def equalize_hist(img):
    L = 256
    M, N = img.shape
    # Compute histogram
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # Compute CDF
    cdf = hist.cumsum()
    # Build LUT: t[k] = round((L-1)/MN * cdf[k])
    t = np.round((L - 1) / (M * N) * cdf).astype(np.uint8)
    # Apply transform
    g = t[img]
    return g, hist, t

# -- Apply custom equalization--------------------------------------------
g_eq, hist_orig, t = equalize_hist(f)

# -- Cross-check with OpenCV equalizeHist---------------------------------
g_cv = cv.equalizeHist(f)

# Histogram of equalized image
hist_eq, _ = np.histogram(g_eq.flatten(), 256, [0, 256])

# -- Display images-------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(g_eq, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Custom Histogram Equalization')

ax[2].imshow(g_cv, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('cv.equalizeHist (cross-check)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q3_images.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Plot histograms------------------------------------------------------
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

ax2[0].bar(np.arange(256), hist_orig, color='gray', width=1)
ax2[0].set_title('Histogram - Original Image')
ax2[0].set_xlabel('Pixel Intensity')
ax2[0].set_ylabel('Count')

ax2[1].bar(np.arange(256), hist_eq, color='gray', width=1)
ax2[1].set_title('Histogram - Equalized Image')
ax2[1].set_xlabel('Pixel Intensity')
ax2[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('output/q3_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
