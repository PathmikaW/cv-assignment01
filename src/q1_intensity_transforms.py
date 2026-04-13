import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the runway image
f = cv.imread('Assignment/runway.png', cv.IMREAD_GRAYSCALE)

# ── (a) Gamma correction  γ = 0.5 ─────────────────────────────────────────
# g = f^γ,  f ∈ [0,1]   (γ < 1 brightens: expands dark pixel range)
gamma = 0.5
t = np.array([(i/255.0)**(gamma)*255 for i in np.arange(0,256)]).astype(np.uint8)
g_05 = cv.LUT(f, t)

# ── (b) Gamma correction  γ = 2.0 ─────────────────────────────────────────
# γ > 1 darkens: compresses dark pixel range
gamma = 2.0
t = np.array([(i/255.0)**(gamma)*255 for i in np.arange(0,256)]).astype(np.uint8)
g_2 = cv.LUT(f, t)

# ── (c) Contrast Stretching  r1=0.2, r2=0.8 ───────────────────────────────
# s(r) = 0             if r < r1
#        (r-r1)/(r2-r1) if r1 <= r <= r2      (r normalised to [0,1])
#        1             if r > r2
r1, r2 = 0.2, 0.8

# Map r1, r2 to integer pixel values
p1 = int(r1 * 255)   # = 51
p2 = int(r2 * 255)   # = 204

t1 = np.linspace(0,   0,   p1).astype(np.uint8)           # r < r1  → 0
t2 = np.linspace(0, 255, p2 - p1 + 1).astype(np.uint8)    # linear ramp
t3 = np.linspace(255, 255, 255 - p2).astype(np.uint8)     # r > r2  → 255

t = np.concatenate((t1, t2, t3)).astype(np.uint8)
g_cs = cv.LUT(f, t)

# ── Display results ────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 4, figsize=(16, 5))

ax[0].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Original Image')

ax[1].imshow(g_05, cmap='gray', vmin=0, vmax=255)
ax[1].set_title('Gamma Correction γ=0.5')

ax[2].imshow(g_2, cmap='gray', vmin=0, vmax=255)
ax[2].set_title('Gamma Correction γ=2')

ax[3].imshow(g_cs, cmap='gray', vmin=0, vmax=255)
ax[3].set_title('Contrast Stretching\nr₁=0.2, r₂=0.8')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.savefig('output/q1_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Plot the three transform functions ─────────────────────────────────────
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))

x = np.arange(0, 256)

ax2[0].plot(x, np.array([(i/255.0)**0.5*255 for i in x]))
ax2[0].set_title('Gamma γ=0.5')
ax2[0].set_xlabel('f(x)')
ax2[0].set_ylabel('g(x)')

ax2[1].plot(x, np.array([(i/255.0)**2.0*255 for i in x]))
ax2[1].set_title('Gamma γ=2')
ax2[1].set_xlabel('f(x)')
ax2[1].set_ylabel('g(x)')

ax2[2].plot(x, t)
ax2[2].set_title('Contrast Stretching\nr₁=0.2, r₂=0.8')
ax2[2].set_xlabel('f(x)')
ax2[2].set_ylabel('g(x)')

plt.tight_layout()
plt.savefig('output/q1_transforms.png', dpi=150, bbox_inches='tight')
plt.show()
