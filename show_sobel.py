import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your folder containing the data files
data_dir = os.path.join(os.getcwd(), 'sobel_results')

# Image dimensions (from sobel_gpu.cpp)
ncols, nrows = 7112, 5146  # width, height

# Build full file paths
input_path = os.path.join(data_dir, 'zebra-gray-int8-4x')
output_path = os.path.join(data_dir, 'processed-raw-int8-4x-cpu.dat')

# 1. Read the original grayscale image (raw bytes)
img = np.fromfile(input_path, dtype=np.uint8)
print("Original image pixels:", img.size)

# 2. Reshape to 2D (rows × columns)
img = img.reshape((nrows, ncols))

# 3. Display the original grayscale image
plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
plt.title("Original Zebra (Grayscale)")
plt.axis('off')
plt.show()

# 4. Read the Sobel-filtered result
sobel_img = np.fromfile(output_path, dtype=np.uint8)
sobel_img = sobel_img.reshape((nrows, ncols))

# 5. Display the Sobel edge detection result
plt.figure(figsize=(10, 8))
plt.imshow(sobel_img, cmap='gray')
plt.title("Sobel Edge Detection Result (CPU)")
plt.axis('off')
plt.show()

# Save both images as PNG for easy viewing
plt.imsave(os.path.join(data_dir, 'zebra_gray.png'), img, cmap='gray')
plt.imsave(os.path.join(data_dir, 'zebra_sobel.png'), sobel_img, cmap='gray')

print("Saved PNG files to:", data_dir)
