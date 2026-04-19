import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


print('Part (a) - Derivation:')
print('  dG/dx = -(x/sigma^2) * G(x,y)')
print('  dG/dy = -(y/sigma^2) * G(x,y)')

# Load image in grayscale
f = cv.imread('Assignment/runway.png', cv.IMREAD_GRAYSCALE)

sigma = 2

# -- (b) Build 5x5 DoG kernels--------------------------------------------
# dG/dx = -x/sigma^2 * G(x,y)
# dG/dy = -y/sigma^2 * G(x,y)
ksize = 5
ax_range = np.arange(-(ksize//2), ksize//2 + 1)
x, y = np.meshgrid(ax_range, ax_range)

G = np.exp(-(x**2 + y**2) / (2 * sigma**2))
DoGx = (-x / sigma**2) * G
DoGy = (-y / sigma**2) * G
DoGx = DoGx / np.abs(DoGx).sum()
DoGy = DoGy / np.abs(DoGy).sum()

# Part (b): Visualise 5x5 DoG kernels as heatmaps
fig_b, ax_b = plt.subplots(1, 2, figsize=(10, 4))
fig_b.suptitle('Part (b): 5x5 Derivative of Gaussian Kernels (sigma=2)', fontsize=13)

im0 = ax_b[0].imshow(DoGx, cmap='RdBu', interpolation='nearest')
ax_b[0].set_title('DoG-x  (dG/dx)')
plt.colorbar(im0, ax=ax_b[0])

im1 = ax_b[1].imshow(DoGy, cmap='RdBu', interpolation='nearest')
ax_b[1].set_title('DoG-y  (dG/dy)')
plt.colorbar(im1, ax=ax_b[1])

plt.tight_layout()
plt.savefig('output/q6b_dog_kernels.png', dpi=150, bbox_inches='tight')
plt.show()

print('Part (b) - 5x5 DoG-x kernel:')
print(np.round(DoGx, 4))
print('\nPart (b) - 5x5 DoG-y kernel:')
print(np.round(DoGy, 4))

# -- (c) Visualise 51x51 DoG-x as 3D surface------------------------------
ksize_big = 51
ax_range_big = np.arange(-(ksize_big//2), ksize_big//2 + 1)
xb, yb = np.meshgrid(ax_range_big, ax_range_big)

Gb = np.exp(-(xb**2 + yb**2) / (2 * sigma**2))
DoGx_big = (-xb / sigma**2) * Gb
DoGx_big = DoGx_big / np.abs(DoGx_big).sum()

fig_c = plt.figure(figsize=(7, 5))
fig_c.suptitle('Part (c): 51x51 DoG-x Surface (sigma=2)', fontsize=13)
ax_c = fig_c.add_subplot(111, projection='3d')
ax_c.plot_surface(xb, yb, DoGx_big, cmap='RdBu')
ax_c.set_xlabel('x')
ax_c.set_ylabel('y')
ax_c.set_zlabel('dG/dx')
plt.tight_layout()
plt.savefig('output/q6c_dog_surface.png', dpi=150, bbox_inches='tight')
plt.show()

# -- (d) Apply DoG kernels-------------------------------------------------
grad_x = cv.filter2D(f, cv.CV_64F, DoGx)
grad_y = cv.filter2D(f, cv.CV_64F, DoGy)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag = cv.normalize(grad_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

fig_d, ax_d = plt.subplots(1, 3, figsize=(15, 5))
fig_d.suptitle('Part (d): DoG Filtering Results', fontsize=13)

ax_d[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax_d[0].set_title('Original Image')

ax_d[1].imshow(np.abs(grad_x), cmap='gray')
ax_d[1].set_title('DoG - x gradient')

ax_d[2].imshow(np.abs(grad_y), cmap='gray')
ax_d[2].set_title('DoG - y gradient')

for a in ax_d:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q6d_dog_gradients.png', dpi=150, bbox_inches='tight')
plt.show()

# -- (e) Compare DoG magnitude vs cv.Sobel--------------------------------
sobel_x = cv.Sobel(f, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(f, cv.CV_64F, 0, 1, ksize=5)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = cv.normalize(sobel_mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

fig_e, ax_e = plt.subplots(1, 2, figsize=(12, 5))
fig_e.suptitle('Part (e): DoG vs Sobel - Gradient Magnitude', fontsize=13)

ax_e[0].imshow(grad_mag, cmap='gray', vmin=0, vmax=255)
ax_e[0].set_title('DoG - Gradient Magnitude')

ax_e[1].imshow(sobel_mag, cmap='gray', vmin=0, vmax=255)
ax_e[1].set_title('Sobel - Gradient Magnitude')

for a in ax_e:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q6e_dog_vs_sobel.png', dpi=150, bbox_inches='tight')
plt.show()
