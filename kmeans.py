import numpy as np
import cv2

def kmeans_segmentation(image, k=3, max_iterations=100, tolerance=1e-5):
    # Flatten the image into a 2D array with shape (height * width, 3)
    image_flat = image.reshape(-1, 3)

    # Initialize the centroids randomly
    centroids = image_flat[np.random.choice(image_flat.shape[0], k, replace=False)]

    # Set previous centroids to None
    prev_centroids = None

    # Iterate until convergence or max iterations
    for i in range(max_iterations):
        # Assign each pixel to the closest centroid
        labels = np.argmin(np.linalg.norm(image_flat[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)

        # Update the centroids as the mean of all pixels assigned to each centroid
        centroids = np.array([np.mean(image_flat[labels == i], axis=0) for i in range(k)])

        # Check for convergence
        if prev_centroids is not None and np.linalg.norm(centroids - prev_centroids) < tolerance:
            break

        # Update the previous centroids
        prev_centroids = centroids

    # Reshape the labels into a 2D array with shape (height, width)
    labels_2d = labels.reshape(image.shape[:2])

    # Map the labels to a color palette
    palette = np.random.randint(0, 256, (k, 3), dtype=np.uint8)
    segmented_image = palette[labels_2d]

    return segmented_image

# Load the input image
image = cv2.imread('samples/image3.png')

# Convert the image to float32 and normalize the pixel values
image = image.astype(np.float32) / 255.0

# Perform K-means segmentation
segmented_image = kmeans_segmentation(image, k=3)

# Display the original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.imwrite('output/kmeans4.jpg',segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()