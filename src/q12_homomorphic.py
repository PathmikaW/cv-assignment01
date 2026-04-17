import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image (non-uniform illumination works best)
f = cv.imread('Assignment/a1images/highlights_and_shadows.jpg', cv.IMREAD_GRAYSCALE)

# ── (a) Multiplicative model: f(x,y) = i(x,y) * r(x,y) ───────────────────
# i = illumination (low-frequency), r = reflectance (high-frequency detail)
# Log transform separates: log(f) = log(i) + log(r)

# ── (b) Log transform ──────────────────────────────────────────────────────
f_log = np.log1p(f.astype(np.float64))   # log(1 + f) avoids log(0)

# ── (c) Homomorphic filtering algorithm ───────────────────────────────────
# 1. Log transform
# 2. FFT
# 3. Apply high-emphasis filter in frequency domain (suppress low, boost high)
# 4. Inverse FFT
# 5. Exp to undo log

# Step 2: FFT of log image
F = np.fft.fft2(f_log)
F_shift = np.fft.fftshift(F)

# Step 3: High-emphasis Gaussian filter in frequency domain
# H(u,v) = (γH - γL)(1 - exp(-D²/(2σ²))) + γL
# γL < 1 attenuates illumination (low freq), γH > 1 boosts reflectance (high freq)
rows, cols = f.shape
u = np.fft.fftshift(np.fft.fftfreq(rows))
v = np.fft.fftshift(np.fft.fftfreq(cols))
V, U = np.meshgrid(v, u)
D2 = U**2 + V**2

gamma_L = 0.25   # attenuate low frequencies (illumination)
gamma_H = 2.0    # boost high frequencies (reflectance/detail)
sigma_freq = 0.1

H = (gamma_H - gamma_L) * (1 - np.exp(-D2 / (2 * sigma_freq**2))) + gamma_L

# Step 4: Apply filter and inverse FFT
G_shift = F_shift * H
G = np.fft.ifftshift(G_shift)
g_log = np.real(np.fft.ifft2(G))

# Step 5: Exp to undo log transform
g = np.expm1(g_log)   # inverse of log1p
g = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# ── (d) Compare with histogram equalization ────────────────────────────────
g_heq = cv.equalizeHist(f)

# ── (e) Display all results ────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Part (e): Homomorphic Filtering vs Histogram Equalization', fontsize=13)

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image\n(non-uniform illumination)')

ax[1].imshow(g, cmap='gray', vmin=0, vmax=255)
ax[1].set_title(f'(c) Homomorphic Filter\n(γL={gamma_L}, γH={gamma_H}, σ={sigma_freq})')

ax[2].imshow(g_heq, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('(d) Histogram Equalization\n(global contrast stretch)')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q12e_homomorphic.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Visualise the frequency-domain filter H ────────────────────────────────
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Part (c): High-Emphasis Filter in Frequency Domain', fontsize=13)

ax2[0].imshow(H, cmap='gray')
ax2[0].set_title('High-Emphasis Filter H(u,v)')
ax2[0].axis('off')

ax2[1].plot(H[rows//2, :])
ax2[1].set_title('1D Cross-section of H(u,v)')
ax2[1].set_xlabel('Frequency')
ax2[1].set_ylabel('Gain')

plt.tight_layout()
plt.savefig('output/q12c_filter.png', dpi=150, bbox_inches='tight')
plt.show()
