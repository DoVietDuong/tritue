from skimage import color, feature, exposure
import cv2
import matplotlib.pyplot as plt

img = plt.imread('hinhnguoi.webp', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('image shape:', img.shape)
print('gray shape: ', gray.shape)

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Gray Image')

plt.show()

# Calculate gradient gx, gy
gx = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=3)

print('gray shape: {}'.format(gray.shape))
print('gx shape: {}'.format(gx.shape))
print('gy shape: {}'.format(gy.shape))

g, theta = cv2.cartToPolar(gx, gy, angleInDegrees=True)
print('gradient format: {}'.format(g.shape))
print('theta format: {}'.format(theta.shape))

w = 20
h = 10

plt.figure(figsize=(w, h))

plt.subplot(1, 4, 1)
plt.imshow(gx, cmap='gray')
plt.title('gradient of x')

plt.subplot(1, 4, 2)
plt.imshow(gy, cmap='gray')
plt.title('gradient of y')

plt.subplot(1, 4, 3)
plt.imshow(g, cmap='gray')
plt.title('Magnitude of gradient')

plt.subplot(1, 4, 4)
plt.imshow(theta, cmap='hsv')
plt.title('Direction of gradient')

plt.show()


print('Original Image Size: ', img.shape)

# 1. Define parameters
cell_size = (8, 8)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins
# 2. Compute parameters passed to HOGDescriptor
winSize = (img.shape[1] // cell_size[1] * cell_size[1],
           img.shape[0] // cell_size[0] * cell_size[0])
blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
blockStride = (cell_size[1], cell_size[0])

print('WinSize (pixels): ', winSize)
print('Block Size (pixels): ', blockSize)
print('Block Stride (pixels): ', blockStride)

# 3. Compute HOG descriptor
hog = cv2.HOGDescriptor(_winSize=winSize,
                        _blockSize=blockSize,
                        _blockStride=blockStride,
                        _cellSize=cell_size,
                        _nbins=nbins)

n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])

# Reshape hog feature
hog_feats = hog.compute(gray.astype('uint8'))\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))

print('HOG Feature Size (h, w, block_size_h, block_size_w, nbins): ', hog_feats.shape)

# Compute HOG
H, hog_image = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
                           visualize=True)

hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
hog_image = hog_image.astype("uint8")

plt.imshow(hog_image, cmap='gray')
plt.title('HOG Image')
plt.show()