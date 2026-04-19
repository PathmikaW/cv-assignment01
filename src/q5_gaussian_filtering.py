import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
f = cv.imread('Assignment/runway.png', cv.IMREAD_GRAYSCALE)

# ── (a) Build 5×5 Gaussian kernel manually ────────────────────────────────
sigma = 2
ksize = 5
ax_range = np.arange(-(ksize//2), ksize//2 + 1)
x, y = np.meshgrid(ax_range, ax_range)
G5 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
G5 = G5 / G5.sum()   # normalise so kernel sums to 1

print('5x5 Gaussian kernel (sigma=2):')
print(np.round(G5, 4))

# ── (b) Build 51×51 Gaussian kernel and visualise as 3D surface ───────────
ksize_big = 51
ax_range_big = np.arange(-(ksize_big//2), ksize_big//2 + 1)
xb, yb = np.meshgrid(ax_range_big, ax_range_big)
G51 = np.exp(-(xb**2 + yb**2) / (2 * sigma**2))
G51 = G51 / G51.sum()

fig_b = plt.figure(figsize=(7, 5))
ax_b = fig_b.add_subplot(111, projection='3d')
ax_b.plot_surface(xb, yb, G51, cmap='viridis')
ax_b.set_title('51×51 Gaussian Kernel (σ=2)')
ax_b.set_xlabel('x')
ax_b.set_ylabel('y')
ax_b.set_zlabel('G(x,y)')
plt.tight_layout()
plt.savefig('output/q5b_gaussian_surface.png', dpi=150, bbox_inches='tight')
plt.show()

# ── (c) Apply manual 5×5 kernel using cv.filter2D ─────────────────────────
g_manual = cv.filter2D(f, -1, G5)

# ── (d) Apply cv.GaussianBlur with same σ and compare ─────────────────────
g_cv = cv.GaussianBlur(f, (ksize, ksize), sigmaX=sigma)

# ── Display: original, manual filter, cv.GaussianBlur ─────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(g_manual, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Manual Gaussian (5×5, σ=2)\ncv.filter2D')

ax[2].imshow(g_cv, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('cv.GaussianBlur (5×5, σ=2)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q5cd_filtered.png', dpi=150, bbox_inches='tight')
plt.show()
