import cv2
import numpy as np
import argparse

def gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    return gradient_magnitude


def apply_threshold(image):
    # Apply binary thresholding
    _, thresh = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    
    # Use thinning operation, approximated with edge detection (Canny or other methods)
    edges = cv2.Canny(thresh, 80, 150)
    
    return edges

def interpolate_points(edges, gradient_magnitude):
    # Find contours to determine points of interest
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty interpolation map
    interpolated = np.zeros_like(gradient_magnitude)
    
    # Iterate over contours to fill the interpolation map
    for c in contours:
        for point in c:
            x, y = point[0]
            interpolated[y, x] = gradient_magnitude[y, x]
    
    # Apply interpolation (can use smoothing techniques like GaussianBlur to approximate interpolation)
    interpolated = cv2.GaussianBlur(interpolated, (21, 21), 0)
    
    return interpolated

def fill_interpolation(interpolated_image, original_image):
    # Fill the image based on interpolated points, making sure the border has higher values to avoid confusion
    filled_image = interpolated_image.copy()
    filled_image[filled_image == 0] = original_image[filled_image == 0]
    
    return filled_image

def validate_segmentation(segmented_image, gradient_magnitude):
    # Label connected components in segmented image
    num_labels, labels_im = cv2.connectedComponents(segmented_image)
    
    validated_image = np.zeros_like(segmented_image)
    
    # Iterate over each labeled component
    for label in range(1, num_labels):  # Skip background label
        mask = np.uint8(labels_im == label)
        edge_strength = cv2.mean(gradient_magnitude, mask=mask)[0]
        
        # Validate: only keep components with strong enough edge strength
        if edge_strength > 50:  # Threshold can be adjusted based on the image
            validated_image[mask > 0] = 255
    
    return validated_image

# Loading Image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())

# Smoothing the image to get a better contour
image = cv2.imread(args["image"])
cv2.imshow("Original Image", image)

# Convert to grayscale and apply Gaussian Blur
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
original = image.copy()
image = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred Image", image)

# Finding Gradient Magnitude
grad_mag = gradient_magnitude(image)
cv2.imshow("Gradient Magnitude", grad_mag)

# Applying Thresholding onto the Gradient Magnitude
edges = apply_threshold(grad_mag)
cv2.imshow("Thresholding", edges)

# Interpolating image using the correct gradient magnitude
interpolated = interpolate_points(edges, grad_mag)
cv2.imshow("Interpolated", interpolated)

# Filling the interpolation with original image data
filled_image = fill_interpolation(interpolated, original)
cv2.imshow("Filled Interpolation", filled_image)

# Segmenting the image
_, segmented = cv2.threshold(filled_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Segmented", segmented)

# Validating the segmentation using gradient magnitude
validated = validate_segmentation(segmented, grad_mag)
cv2.imshow("Validated", validated)
cv2.waitKey(0)
