# IT5437 Computer Vision — Assignment 1

Intensity Transformations and Neighbourhood Filtering
**MSc in Artificial Intelligence, University of Moratuwa** | [GitHub](https://github.com/PathmikaW/cv-assignment01)

## Questions

| # | Topic |
|---|-------|
| Q1 | Gamma correction and contrast stretching |
| Q2 | Gamma correction in L*a*b* colour space |
| Q3 | Custom histogram equalisation |
| Q4 | Otsu thresholding and selective foreground equalisation |
| Q5 | Manual Gaussian kernel and 3D surface visualisation |
| Q6 | Derivative of Gaussian kernels and gradient magnitude |
| Q7 | Image zoom — nearest-neighbour vs bilinear interpolation |
| Q8 | Salt-and-pepper noise filtering — Gaussian vs median |
| Q9 | Image sharpening via unsharp masking |
| Q10 | Manual bilateral filter |
| Q11 | FFT frequency response analysis |
| Q12 | Homomorphic filtering |

## Structure

```
src/        one script per question (q1 to q12)
output/     generated figures (32 PNG files)
Assignment/ input images
```

## Setup

```bash
uv sync
```

## Run

```bash
uv run python src/q1_intensity_transforms.py
uv run python src/q2_lab_gamma.py
uv run python src/q3_hist_equalization.py
uv run python src/q4_otsu_foreground.py
uv run python src/q5_gaussian_filtering.py
uv run python src/q6_derivative_of_gaussian.py
uv run python src/q7_image_zoom.py
uv run python src/q8_noise_filtering.py
uv run python src/q9_sharpening.py
uv run python src/q10_bilateral_filter.py
uv run python src/q11_frequency_response.py
uv run python src/q12_homomorphic.py
```

Output images are saved to `output/`.
