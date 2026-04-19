import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the woman-at-door image, convert to grayscale
img = cv.imread('Assignment/a1images/Woman standing in front of an open door.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# -- (a) Otsu Thresholding------------------------------------------------
# Automatically finds optimal threshold that minimises intra-class variance
ret, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f'Otsu threshold value: {ret}')

# -- (b) Selective Histogram Equalization (foreground only)---------------
# Extract foreground pixels (where mask == 255)
foreground_pixels = gray[mask == 255]

# Build equalization LUT from foreground pixels only
L = 256
n_fg = foreground_pixels.size
hist, _ = np.histogram(foreground_pixels, 256, [0, 256])
cdf = hist.cumsum()
t = np.round((L - 1) / n_fg * cdf).astype(np.uint8)

# Apply LUT to full image, then restore only foreground
g_eq = gray.copy()
g_eq[mask == 255] = t[gray[mask == 255]]

# -- Display results-------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(18, 5))

ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

ax[1].imshow(gray, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Grayscale')

ax[2].imshow(mask, cmap='gray', vmin=0, vmax=255)
ax[2].set_title(f'Otsu Mask\n(threshold = {int(ret)})')

ax[3].imshow(g_eq, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('Foreground Equalized\n(hidden details revealed)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q4a_otsu_mask.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Plot histograms: foreground before and after equalization------------
hist_after, _ = np.histogram(g_eq[mask == 255], 256, [0, 256])

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

ax2[0].bar(np.arange(256), hist, color='gray', width=1)
ax2[0].set_title('Foreground Histogram - Before Equalization')
ax2[0].set_xlabel('Pixel Intensity')
ax2[0].set_ylabel('Count')

ax2[1].bar(np.arange(256), hist_after, color='gray', width=1)
ax2[1].set_title('Foreground Histogram - After Equalization')
ax2[1].set_xlabel('Pixel Intensity')
ax2[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('output/q4b_foreground_equalized.png', dpi=150, bbox_inches='tight')
plt.show()
