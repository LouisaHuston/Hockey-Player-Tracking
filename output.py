import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Example detections list with multiple bounding boxes for a single image
detections = [
    {
        "image_id": 1,
        "bounding_boxes": [
            {"bbox": [430, 278, 132, 169], "score": 0.85, "category_id": 1, "category_name": "player", "db_id": 123},
            {"bbox": [1200, 350, 100, 150], "score": 0.92, "category_id": 2, "category_name": "puck", "db_id": 124}
        ]
    },
    # Add more dictionaries here for additional frames/images
]

# Folder to save the output images
output_folder = '/content/output_images'
os.makedirs(output_folder, exist_ok=True)

# Step 1: Load the blank rink image (1232x637)
blank_image_path = '/content/unnamed.jpg'  # Path to your blank rink image
blank_image = cv2.imread(blank_image_path)

# Ensure the image is loaded successfully
if blank_image is None:
    print("Error: Could not load the blank rink image.")
    exit()

# Step 2: Take text input for the real-world coordinates (in feet) for the 4 points
real_coords = []
print("Please enter the real-world coordinates (in feet) for each of the 4 points:")

for i in range(4):
    x_real = float(input(f"Enter real-world x coordinate for point {i+1} (in feet): "))
    y_real = float(input(f"Enter real-world y coordinate for point {i+1} (in feet): "))
    real_coords.append([x_real, y_real])

real_coords = np.array(real_coords, dtype='float32')

# Convert pixel_coords and real_coords to the proper shape for cv2.findHomography
pixel_coords = np.array([], dtype='float32').reshape(-1, 1, 2)  # Initialize as an empty array for homography calculation
real_coords = real_coords.reshape(-1, 1, 2)

# Step 3: Calculate the homography matrix
homography_matrix, _ = cv2.findHomography(pixel_coords, real_coords)

# Step 4: Process each frame's detections
for detection in detections:
    # Copy the blank image for each frame
    image_copy = blank_image.copy()
    
    # Extract the bounding boxes for this image
    bounding_boxes = detection["bounding_boxes"]
    
    # Iterate over each bounding box
    for bbox_info in bounding_boxes:
        # Extract bounding box information
        bbox = bbox_info["bbox"]  # [x1, y1, width, height]
        x1, y1, width, height = bbox
        
        # Calculate the bottom middle of the bounding box
        bbox_center_x = x1 + width // 2  # Horizontal center of the bounding box
        bbox_center_y = y1 + height // 2  # Vertical center of the bounding box
        
        # Convert pixel coordinates to real-world coordinates (feet) using the homography matrix
        bbox_pixel_coords = np.array([bbox_center_x, bbox_center_y], dtype='float32').reshape(1, 1, 2)
        real_coords_at_bottom_middle = cv2.perspectiveTransform(bbox_pixel_coords, homography_matrix)
        real_coords_at_bottom_middle = real_coords_at_bottom_middle[0][0]

        # Convert real-world coordinates (feet) to pixel coordinates on the rink image
        real_rink_width = 200  # real width in feet
        real_rink_height = 85  # real height in feet

        # Dimensions of the blank image (in pixels)
        image_width, image_height = 1232, 637  # size of the blank rink image

        # Convert real coordinates to pixel coordinates
        scaled_pixel_x = (real_coords_at_bottom_middle[0] / real_rink_width) * image_width
        scaled_pixel_y = (real_coords_at_bottom_middle[1] / real_rink_height) * image_height

        # Draw a small black circle at the calculated pixel coordinates for each bounding box
        circle_radius = 5  # Radius of the circle
        cv2.circle(image_copy, (int(scaled_pixel_x), int(scaled_pixel_y)), circle_radius, (0, 0, 0), -1)  # Black circle

    # Step 5: Save the image with the circles drawn
    output_image_path = os.path.join(output_folder, f"frame_{detection['image_id']}_with_dots.jpg")
    cv2.imwrite(output_image_path, image_copy)

# Step 6: Convert images in output_folder into a video
video_output_path = '/content/bounding_boxes_video.avi'  # Path to save the video
image_files = sorted(os.listdir(output_folder), key=lambda x: int(x.split('_')[1]))  # Sort images by image_id

# Ensure there are images to process
if len(image_files) == 0:
    print("Error: No images found in the output folder.")
    exit()

# Read the first image to get the frame dimensions (height, width)
first_image_path = os.path.join(output_folder, image_files[0])
first_image = cv2.imread(first_image_path)
frame_height, frame_width, _ = first_image.shape

# Initialize the VideoWriter to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec to use for the video
fps = 30  # Frames per second (adjust this as needed)
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

# Read each image and add it to the video
for image_file in image_files:
    image_path = os.path.join(output_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_file}. Skipping.")
        continue
    
    # Write the current image to the video
    video_writer.write(image)

# Release the VideoWriter and finalize the video file
video_writer.release()

# Print confirmation message after the video is created
print(f"Video saved to: {video_output_path}")

# # Code works for manually inputting bounding box coordinates
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# # Step 1: Manually input the real coordinates and the bounding box information
# # Example: Provide real-world coordinates in feet for the 4 points
# real_coords = np.array([
#     [10, 40.5],  # Point 1
#     [10, 44.5],  # Point 2
#     [20, 63],    # Point 3
#     [70, 0]      # Point 4
# ], dtype='float32')

# # Manually input the bounding box details (bounding box coordinates and size in pixels)
# bbox_x = 430  # X coordinate of the top-left corner of the bounding box
# bbox_y = 278  # Y coordinate of the top-left corner of the bounding box
# bbox_width = 132  # Width of the bounding box in pixels
# bbox_height = 169  # Height of the bounding box in pixels

# # Step 2: Define pixel coordinates for the 4 points that correspond to real_coords
# # For simplicity, you can directly input the pixel coordinates manually
# pixel_coords = np.array([
#     [309, 447],  # Pixel coordinates for point 1
#     [353, 426],  # Pixel coordinates for point 2
#     [1424, 359], # Pixel coordinates for point 3
#     [1427, 571]  # Pixel coordinates for point 4
# ], dtype='float32')

# # Reshape the arrays to be 2D (4, 2) for the homography function
# pixel_coords = pixel_coords.reshape(-1, 1, 2)  # Shape should be (4, 1, 2)
# real_coords = real_coords.reshape(-1, 1, 2)    # Shape should be (4, 1, 2)

# # Flatten real_coords to 2D (4, 2) to allow access to columns directly
# real_coords_flat = real_coords.reshape(-1, 2)

# # Step 3: Calculate the homography matrix
# homography_matrix, _ = cv2.findHomography(pixel_coords, real_coords_flat)

# # Step 4: Manually define the pixel coordinates of the bottom middle of the bounding box
# # The bottom middle of the bounding box in pixel coordinates
# bbox_center_x = bbox_x + bbox_width // 2  # Horizontal center of the bounding box
# bbox_center_y = bbox_y + bbox_height // 2  # Vertical center of the bounding box
# bbox_bottom_middle_pixel = np.array([bbox_center_x, bbox_center_y + bbox_height // 2], dtype='float32').reshape(1, 1, 2)

# # Step 5: Apply the homography to convert to real-world coordinates in feet
# real_coords_at_bottom_middle = cv2.perspectiveTransform(bbox_bottom_middle_pixel, homography_matrix)
# real_coords_at_bottom_middle = real_coords_at_bottom_middle[0][0]

# # Step 6: Convert real-world coordinates (feet) to pixel coordinates on the rink image
# # Dimensions of the real rink (in feet)
# real_rink_width = 200  # real width in feet
# real_rink_height = 85  # real height in feet

# # Dimensions of the blank image (in pixels)
# image_width, image_height = 1232, 637  # size of the blank rink image

# # Convert real coordinates to pixel coordinates
# scaled_pixel_x = (real_coords_at_bottom_middle[0] / real_rink_width) * image_width
# scaled_pixel_y = (real_coords_at_bottom_middle[1] / real_rink_height) * image_height

# # Final pixel coordinates
# final_pixel_x, final_pixel_y = scaled_pixel_x, scaled_pixel_y

# # Print the results
# print(f"Real coordinates (in feet): {real_coords_at_bottom_middle}")
# print(f"Scaled pixel coordinates on new image: ({final_pixel_x}, {final_pixel_y})")

# # Step 7: Load the pre-stored blank rink image (1232x637)
# blank_image_path = 'assets/blankrink.jpg'  # Path to your blank rink image
# blank_image = cv2.imread(blank_image_path)

# # Ensure the image is loaded successfully
# if blank_image is None:
#     print("Error: Could not load the blank rink image.")
#     exit()

# # Step 8: Draw a small black circle at the final pixel coordinates on the blank image
# circle_radius = 5  # Radius of the circle
# cv2.circle(blank_image, (int(final_pixel_x), int(final_pixel_y)), circle_radius, (0, 0, 0), -1)  # Black circle

# # Step 9: Save the image with the circle drawn
# output_image_path = '/content/final_image_with_circle.jpg'  # Path to save the image
# cv2.imwrite(output_image_path, blank_image)

# # Show the final image with the circle
# final_image_rgb = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
# plt.imshow(final_image_rgb)
# plt.axis('off')  # Turn off axis for clean display
# plt.show()
