import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ── Helper: compute centered log-magnitude FFT of a kernel ────────────────
def fft_magnitude(kernel, size=256):
    # Zero-pad kernel to size×size for fine frequency resolution
    h = np.zeros((size, size))
    kH, kW = kernel.shape
    cy, cx = size // 2, size // 2
    h[cy - kH//2 : cy + kH//2 + 1, cx - kW//2 : cx + kW//2 + 1] = kernel
    H = np.fft.fftshift(np.fft.fft2(h))
    return np.log(1 + np.abs(H))


# ── (b) Define the three kernels ───────────────────────────────────────────

# Box (averaging) filter — 9×9
box = np.ones((9, 9), dtype=np.float64) / 81

# Gaussian filter — 9×9, σ=2
sigma = 2
k = 9
ax_range = np.arange(-(k//2), k//2 + 1)
x, y = np.meshgrid(ax_range, ax_range)
gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
gauss = gauss / gauss.sum()

# Laplacian filter (edge detector / high-pass)
laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float64)

# ── (b) Compute FFT magnitudes ─────────────────────────────────────────────
fft_box  = fft_magnitude(box)
fft_gauss = fft_magnitude(gauss)
fft_lap  = fft_magnitude(laplacian)

# ── (b) Plot frequency responses ───────────────────────────────────────────
fig_b, ax_b = plt.subplots(1, 3, figsize=(15, 5))
fig_b.suptitle('Part (b): Frequency Response (log magnitude) of Filters', fontsize=13)

ax_b[0].imshow(fft_box, cmap='gray')
ax_b[0].set_title('Box (Averaging) Filter\n→ Low-pass with strong sidelobes')

ax_b[1].imshow(fft_gauss, cmap='gray')
ax_b[1].set_title('Gaussian Filter\n→ Smooth low-pass, no ringing')

ax_b[2].imshow(fft_lap, cmap='gray')
ax_b[2].set_title('Laplacian Filter\n→ High-pass (boosts edges)')

for a in ax_b:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q11b_frequency_response.png', dpi=150, bbox_inches='tight')
plt.show()

# ── (c) 1D cross-section: Gaussian vs ideal low-pass (box) ────────────────
# Shows smooth rolloff of Gaussian vs abrupt cutoff causing ringing
fft_box_1d  = fft_magnitude(box)[128, :]
fft_gauss_1d = fft_magnitude(gauss)[128, :]

fig_c, ax_c = plt.subplots(figsize=(10, 4))
fig_c.suptitle('Part (c): 1D Frequency Profile — Gaussian vs Box Filter', fontsize=13)
ax_c.plot(fft_box_1d,  label='Box filter (abrupt cutoff → ringing)')
ax_c.plot(fft_gauss_1d, label='Gaussian filter (smooth rolloff → no ringing)')
ax_c.set_xlabel('Frequency (pixels)')
ax_c.set_ylabel('Log magnitude')
ax_c.legend()
plt.tight_layout()
plt.savefig('output/q11c_gaussian_vs_box.png', dpi=150, bbox_inches='tight')
plt.show()

# ── (d) Apply filters to a noisy image and compare ────────────────────────
f = cv.imread('Assignment/a1images/emma.jpg', cv.IMREAD_GRAYSCALE)

# Add Gaussian noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 25, f.shape).astype(np.float64)
f_noisy = np.clip(f.astype(np.float64) + noise, 0, 255).astype(np.uint8)

g_box   = cv.filter2D(f_noisy, -1, box.astype(np.float32))
g_gauss = cv.GaussianBlur(f_noisy, (9, 9), sigma)
g_lap   = cv.filter2D(f_noisy, -1, laplacian.astype(np.float32))

fig_d, ax_d = plt.subplots(1, 5, figsize=(22, 5))
fig_d.suptitle('Part (d): Filter Comparison on Noisy Image', fontsize=13)

ax_d[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax_d[0].set_title('Original')

ax_d[1].imshow(f_noisy, cmap='gray', vmin=0, vmax=255)
ax_d[1].set_title('Noisy Image')

ax_d[2].imshow(g_box, cmap='gray', vmin=0, vmax=255)
ax_d[2].set_title('Box Filter\n(blurs + ringing)')

ax_d[3].imshow(g_gauss, cmap='gray', vmin=0, vmax=255)
ax_d[3].set_title('Gaussian Filter\n(best noise reduction)')

ax_d[4].imshow(np.clip(g_lap + 128, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
ax_d[4].set_title('Laplacian Filter\n(amplifies noise)')

for a in ax_d:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q11d_filter_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
