import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save them as images in the specified output folder.

    Args:
    - video_path (str): Path to the source video file.
    - output_folder (str): Directory where extracted frames will be saved.
    - frame_rate (int): Number of frames to extract per second. Default is 1.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames in Video: {total_frames}")
    
    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Frames Per Second: {fps}")

    # Calculate the interval between frames to be saved
    # Ensures at least one frame is extracted every second if frame_rate = 1
    frame_interval = max(int(fps / frame_rate), 1)
    print(f"Frame Interval: {frame_interval}")

    # Frame counter
    frame_number = 0
    
    # Read and extract frames from the video
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # Break the loop if no frame is returned (end of video)
        if not ret:
            break
        
        # Save every nth frame based on the calculated interval
        if frame_number % frame_interval == 0:
            # Construct the filename for the saved frame
            frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
            
            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
        
        # Increment the frame counter
        frame_number += 1
    
    # Release the video capture object
    video.release()
    print("Frame extraction complete.")

# Example usage
video_path = "Input/1.mp4"  # Input your video file path here
output_folder = "D:\Yugam\VS Code\SIH\Output"  # Input your desired output folder here
frame_rate = 1  # Number of frames to extract per second; adjust as needed

# Call the function to extract frames
extract_frames(video_path, output_folder, frame_rate)
