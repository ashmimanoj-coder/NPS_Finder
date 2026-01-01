import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# ---------------------------
# Load Image
# ---------------------------
path_input = input("Please enter the path of your image: ")
image_path = path_input.strip().replace('"','')

if os.path.exists(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        print("success!Running Combined Analysis...")
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        noise_residual = cv2.absdiff(img, blur)
        noise_residual = cv2.normalize(noise_residual,None,0,255,cv2.NORM_MINMAX)
        temp_file = 'analysis_temp.jpg'
        cv2.imwrite(temp_file, img,[cv2.IMWRITE_JPEG_QUALITY, 90])
        temp_img = cv2.imread(temp_file)
        pattern_diff = cv2.absdiff(img, temp_img)
        pattern_analysis = cv2.multiply(pattern_diff,15)
        h,w, _ = img.shape
        print(f"Image dimensions: {w}x{h}")
        cv2.imshow("1.Original image",img)
        cv2.imshow("2.Noise residual (Sensor Noise)",noise_residual)
        cv2.imshow("3.Pixel Pattern(Compression ELA",pattern_analysis)
        print("Analysis Completed")
        print("Window 2 shows semsor/grain noise.")
        print("Window 3 shows editing/compression inconsistencies.")
        print("Press any key on your keyboard to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error:OpenCV could not decode the image.")
else:
    print("Error:Image not found.")


# ---------------------------
# 1. Noise Residual Analysis
# ---------------------------
blur = cv2.GaussianBlur( img, (5, 5), 0)
noise_residual = img - blur
noise_level = np.std(noise_residual)

# ---------------------------
# 2. Signal-to-Noise Ratio
# ---------------------------
signal_power = np.mean(img ** 2)
noise_power = np.mean(noise_residual ** 2)
snr = 10 * np.log10(signal_power / noise_power )

# ---------------------------
# 3. Pixel Intensity Histogram
# ---------------------------
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# ---------------------------
# 4. Pixel Difference Pattern
# ---------------------------
dx = np.abs(np.diff(img, axis=1))
dy = np.abs(np.diff(img, axis=0))

# ---------------------------
# 5. PRNU-style Noise Pattern
# ---------------------------
denoised = cv2.fastNlMeansDenoising(img, None, 7, 21)
noise_pattern = img - denoised

# ---------------------------
# 6. Frequency Domain Analysis (FFT)
# ---------------------------
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)
fft_magnitude = 20 * np.log(np.abs(fft_shift) + 1)

# ---------------------------
# 7. Block-based Pixel Pattern
# ---------------------------
block_size = 8
h, w, _= img.shape
block_means = []

for y in range(0, h - block_size, block_size):
    for x in range(0, w - block_size, block_size):
        block = img[y:y+block_size, x:x+block_size]
        block_means.append(np.mean(block))

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(14,10))

plt.subplot(3,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(3,3,2)
plt.title("Noise Residual")
plt.imshow(noise_residual, cmap='gray')

plt.subplot(3,3,3)
plt.title("Noise Pattern (PRNU)")
plt.imshow(noise_pattern, cmap='gray')

plt.subplot(3,3,4)
plt.title("Histogram")
plt.plot(hist)

plt.subplot(3,3,5)
plt.title("Horizontal Differences")
plt.imshow(dx, cmap='gray')

plt.subplot(3,3,6)
plt.title("Vertical Differences")
plt.imshow(dy, cmap='gray')

plt.subplot(3,3,7)
plt.title("FFT Spectrum")
plt.imshow(fft_magnitude, cmap='gray')

plt.subplot(3,3,8)
plt.title("Block Mean Pattern")
plt.plot(block_means)

plt.tight_layout()
plt.show()

# ---------------------------
# Results
# ---------------------------
print("Estimated Noise Level:", noise_level)
print("SNR (dB):", snr)
