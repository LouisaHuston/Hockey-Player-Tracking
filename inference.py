def main():

  # 1) Extract frames from a video path you specify
  def extract_frames_from_video(video_path, output_folder):
    import cv2
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

  video_path = input("Enter the path to the video file: ")
  frames_output_folder = 'extracted_frames'
  extract_frames_from_video(video_path, frames_output_folder)

  # 2) Specify video relative location to the rink
  import cv2
  import numpy as np
  
  def calculate_homography(frame_points, rink_points):
      """
      Calculates the homography matrix for perspective transformation.
  
      :param frame_points: 2D coordinates from the video frame (e.g., corners of the rink in the frame)
      :param rink_points: 2D real-world coordinates (e.g., corresponding corners of the rink on the actual surface)
      :return: homography matrix
      """
      frame_points = np.float32(frame_points)
      rink_points = np.float32(rink_points)
  
      # Calculate the homography matrix
      homography_matrix, status = cv2.findHomography(frame_points, rink_points)
      return homography_matrix

  # 3) Run inference on those frames - save result in COCO format to data/annoations/inference.json
  
  # 4) Use data/annoations/inference.json to predict locations for each of the bboxes for each of the frames

  # 5) Use those predicted lcoations to identify where each person is on the rink - make a picture for each frame 

  # 6) Turn all those frames into a movie

  pass


if __name__ == '__main__':
  main()
