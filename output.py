import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Manually input the real coordinates and the bounding box information
# Example: Provide real-world coordinates in feet for the 4 points
real_coords = np.array([
    [10, 40.5],  # Point 1
    [10, 44.5],  # Point 2
    [20, 63],    # Point 3
    [70, 0]      # Point 4
], dtype='float32')

# Manually input the bounding box details (bounding box coordinates and size in pixels)
bbox_x = 430  # X coordinate of the top-left corner of the bounding box
bbox_y = 278  # Y coordinate of the top-left corner of the bounding box
bbox_width = 132  # Width of the bounding box in pixels
bbox_height = 169  # Height of the bounding box in pixels

# Step 2: Define pixel coordinates for the 4 points that correspond to real_coords
# For simplicity, you can directly input the pixel coordinates manually
pixel_coords = np.array([
    [309, 447],  # Pixel coordinates for point 1
    [353, 426],  # Pixel coordinates for point 2
    [1424, 359], # Pixel coordinates for point 3
    [1427, 571]  # Pixel coordinates for point 4
], dtype='float32')

# Reshape the arrays to be 2D (4, 2) for the homography function
pixel_coords = pixel_coords.reshape(-1, 1, 2)  # Shape should be (4, 1, 2)
real_coords = real_coords.reshape(-1, 1, 2)    # Shape should be (4, 1, 2)

# Flatten real_coords to 2D (4, 2) to allow access to columns directly
real_coords_flat = real_coords.reshape(-1, 2)

# Step 3: Calculate the homography matrix
homography_matrix, _ = cv2.findHomography(pixel_coords, real_coords_flat)

# Step 4: Manually define the pixel coordinates of the bottom middle of the bounding box
# The bottom middle of the bounding box in pixel coordinates
bbox_center_x = bbox_x + bbox_width // 2  # Horizontal center of the bounding box
bbox_center_y = bbox_y + bbox_height // 2  # Vertical center of the bounding box
bbox_bottom_middle_pixel = np.array([bbox_center_x, bbox_center_y + bbox_height // 2], dtype='float32').reshape(1, 1, 2)

# Step 5: Apply the homography to convert to real-world coordinates in feet
real_coords_at_bottom_middle = cv2.perspectiveTransform(bbox_bottom_middle_pixel, homography_matrix)
real_coords_at_bottom_middle = real_coords_at_bottom_middle[0][0]

# Step 6: Convert real-world coordinates (feet) to pixel coordinates on the rink image
# Dimensions of the real rink (in feet)
real_rink_width = 200  # real width in feet
real_rink_height = 85  # real height in feet

# Dimensions of the blank image (in pixels)
image_width, image_height = 1232, 637  # size of the blank rink image

# Convert real coordinates to pixel coordinates
scaled_pixel_x = (real_coords_at_bottom_middle[0] / real_rink_width) * image_width
scaled_pixel_y = (real_coords_at_bottom_middle[1] / real_rink_height) * image_height

# Final pixel coordinates
final_pixel_x, final_pixel_y = scaled_pixel_x, scaled_pixel_y

# Print the results
print(f"Real coordinates (in feet): {real_coords_at_bottom_middle}")
print(f"Scaled pixel coordinates on new image: ({final_pixel_x}, {final_pixel_y})")

# Step 7: Load the pre-stored blank rink image (1232x637)
blank_image_path = '/content/unnamed.jpg'  # Path to your blank rink image
blank_image = cv2.imread(blank_image_path)

# Ensure the image is loaded successfully
if blank_image is None:
    print("Error: Could not load the blank rink image.")
    exit()

# Step 8: Draw a small black circle at the final pixel coordinates on the blank image
circle_radius = 5  # Radius of the circle
cv2.circle(blank_image, (int(final_pixel_x), int(final_pixel_y)), circle_radius, (0, 0, 0), -1)  # Black circle

# Step 9: Save the image with the circle drawn
output_image_path = '/content/final_image_with_circle.jpg'  # Path to save the image
cv2.imwrite(output_image_path, blank_image)

# Show the final image with the circle
final_image_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
plt.imshow(final_image_rgb)
plt.axis('off')  # Turn off axis for clean display
plt.show()
