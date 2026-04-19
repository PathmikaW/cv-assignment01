import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
f = cv.imread('Assignment/a1images/daisy.jpg', cv.IMREAD_GRAYSCALE)

# -- Unsharp masking: sharp = f + alpha * (f - blur(f))-------------------
# f - blur(f) is the high-frequency detail (mask)
# alpha controls sharpening strength

blur = cv.GaussianBlur(f, (5, 5), 2)
mask = f.astype(np.float64) - blur.astype(np.float64)

alpha = 1.5
sharp = np.clip(f.astype(np.float64) + alpha * mask, 0, 255).astype(np.uint8)

# -- Display---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Image Sharpening via Unsharp Masking (alpha=1.5)', fontsize=13)

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(np.clip(mask + 128, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Unsharp Mask\n(high-frequency detail)')

ax[2].imshow(sharp, cmap='gray', vmin=0, vmax=255)
ax[2].set_title(f'Sharpened Image\n(alpha={alpha})')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q9_sharpening.png', dpi=150, bbox_inches='tight')
plt.show()
