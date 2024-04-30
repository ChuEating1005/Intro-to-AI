import cv2
import numpy as np

image = cv2.imread('data/image.png')

# Translation
tx, ty = 100, 100
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
cv2.imwrite('image/translated_image.png', translated_image)

# Rotation
angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
cv2.imwrite('image/rotated_image.png', rotated_image)

# Flipping
flipped_image_horizontally = cv2.flip(image, 1)
cv2.imwrite('image/flipped_image_horizontally.png', flipped_image_horizontally)

flipped_image_vertically = cv2.flip(image, 0)
cv2.imwrite('image/flipped_image_vertically.png', flipped_image_vertically)

# Scaling
fx, fy = 0.5, 0.5
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('image/scaled_image.png', scaled_image)

# Cropping
start_row, start_col = 10, 10
end_row, end_col = 100, 100
cropped_image = image[start_row:end_row, start_col:end_col]
cv2.imwrite('image/cropped_image.png', cropped_image)