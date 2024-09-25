import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# Load the Haar Cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Convert BGR image to RGB
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    return faces, img_rgb

def save_cropped_faces(faces, img, output_folder, base_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, (x, y, w, h) in enumerate(faces):
        # Crop the face from the image
        face_img = img[y:y+h, x:x+w]
        
        # Save the cropped face image
        face_filename = f"{os.path.splitext(base_filename)[0]}_face_{i}.png"
        face_path = os.path.join(output_folder, face_filename)
        cv.imwrite(face_path, face_img)

def save_processed_image_with_annotations(img_rgb, faces, output_folder, image_name):
    # Convert RGB image back to BGR for saving
    img_annotated = img_rgb.copy()

    # Convert RGB to BGR for OpenCV
    img_bgr = cv.cvtColor(img_annotated, cv.COLOR_RGB2BGR)

    # Draw rectangles and coordinates on the image
    for (x, y, w, h) in faces:
        cv.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img_bgr, f'({x},{y})', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    output_path = os.path.join(output_folder, image_name)
    cv.imwrite(output_path, img_bgr)

def process_frames(video_path, image_folder, output_folder_faces, output_folder_processed):
    # Ensure output folders exist
    if not os.path.exists(output_folder_faces):
        os.makedirs(output_folder_faces)
    if not os.path.exists(output_folder_processed):
        os.makedirs(output_folder_processed)

    # Open the video file
    cap = cv.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)  # Interval of 0.5 seconds

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            image_name = f"{timestamp}.jpg"
            image_path = os.path.join(image_folder, image_name)
            cv.imwrite(image_path, frame)
            print(f"Saved frame at {timestamp}")

            # Detect faces in the frame
            faces, img_rgb = detect_faces(frame)

            # Save cropped face images
            save_cropped_faces(faces, frame, output_folder_faces, image_name)

            # Save the processed image with detected faces and annotations
            save_processed_image_with_annotations(img_rgb, faces, output_folder_processed, image_name)

            # Display the image with Matplotlib
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img_rgb)

            for (x, y, w, h) in faces:
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                # Add text with coordinates
                ax.text(x, y - 10, f'({x},{y})', color='green', fontsize=12, weight='bold')

            plt.title(f'File: {image_name}')
            plt.axis('off')  # Hide axes
            plt.show()

        frame_count += 1

    cap.release()
    print("Done")

if __name__ == "__main__":
    video_path = r'1.mp4'  # Use raw string for Windows paths
    image_folder = r'facial_frames'  # Folder to save extracted frames
    output_folder_faces = r'output_faces'  # Folder to save cropped faces
    output_folder_processed = r'outputs_processed'  # Folder to save processed images with annotations

    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
    else:
        process_frames(video_path, image_folder, output_folder_faces, output_folder_processed)
