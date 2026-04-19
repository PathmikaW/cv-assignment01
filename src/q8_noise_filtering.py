import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
f = cv.imread('Assignment/a1images/emma.jpg', cv.IMREAD_GRAYSCALE)

# -- Add salt & pepper noise-----------------------------------------------
def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(img.shape)
    noisy[mask < prob / 2] = 0          # pepper
    noisy[mask > 1 - prob / 2] = 255    # salt
    return noisy

noisy = add_salt_pepper(f, prob=0.05)

# -- (a) Gaussian smoothing------------------------------------------------
g_gaussian = cv.GaussianBlur(noisy, (5, 5), 1)

# -- (b) Median filtering-------------------------------------------------
g_median = cv.medianBlur(noisy, 5)

# -- Display: original, noisy, gaussian, median----------------------------
fig, ax = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Part (a) & (b): Salt & Pepper Noise Filtering', fontsize=13)

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Salt & Pepper Noise\n(prob=0.05)')

ax[2].imshow(g_gaussian, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('(a) Gaussian Blur\n(5x5, sigma=1)')

ax[3].imshow(g_median, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('(b) Median Filter\n(5x5)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q8ab_noise_filtering.png', dpi=150, bbox_inches='tight')
plt.show()

# -- Quantitative comparison (MSE vs original)-----------------------------
def mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

print(f'MSE - Noisy vs Original:    {mse(noisy, f):.2f}')
print(f'MSE - Gaussian vs Original: {mse(g_gaussian, f):.2f}')
print(f'MSE - Median vs Original:   {mse(g_median, f):.2f}')
