import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the highlights and shadows image (color)
img = cv.imread('Assignment/a1images/highlights_and_shadows.jpg')

# ── Convert BGR → L*a*b* ───────────────────────────────────────────────────
# OpenCV encodes L in [0,255] (L* is [0,100] in standard, scaled here)
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
L, a, b = cv.split(lab)

# ── (a) Apply gamma to L channel ──────────────────────────────────────────
# γ = 0.5 brightens dark regions (image appears underexposed → apply γ < 1)
gamma = 0.5
t = np.array([(i/255.0)**(gamma)*255 for i in np.arange(0,256)]).astype(np.uint8)
L_corrected = cv.LUT(L, t)

# Merge corrected L back with original a, b channels
lab_corrected = cv.merge([L_corrected, a, b])
img_corrected = cv.cvtColor(lab_corrected, cv.COLOR_Lab2BGR)

# ── (a) Display original and gamma-corrected images ───────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

ax[1].imshow(cv.cvtColor(img_corrected, cv.COLOR_BGR2RGB))
ax[1].set_title(f'Gamma Corrected (γ={gamma})\nApplied to L* channel only')

for a_ in ax:
    a_.axis('off')

plt.tight_layout()
plt.savefig('output/q2_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ── (b) Plot histograms of original L and corrected L ─────────────────────
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

ax2[0].hist(L.flatten(), bins=256, range=[0, 256], color='gray')
ax2[0].set_title('Histogram of Original L* Channel')
ax2[0].set_xlabel('Pixel Intensity')
ax2[0].set_ylabel('Count')

ax2[1].hist(L_corrected.flatten(), bins=256, range=[0, 256], color='gray')
ax2[1].set_title(f'Histogram of Corrected L* Channel (γ={gamma})')
ax2[1].set_xlabel('Pixel Intensity')
ax2[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('output/q2_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
