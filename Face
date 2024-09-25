from facenet_pyfrom facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm
import mtcnn
import matplotlib.pyplot as plt

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define our MTCNN extractor
fast_mtcnn = MTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()

    for filename in tqdm(filenames):
        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):
            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:
                faces = fast_mtcnn(frames)
                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []

                print(
                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )
        v_cap.stop()

# Example filenames for detection
filenames = ['video1.mp4', 'video2.mp4']  # Replace with actual filenames
run_detection(fast_mtcnn, filenames)

# Print version of mtcnn
print(mtcnn._version_)

# Load image from file and display
filename = "glediston-bastos-ZtmmR9D_2tA-unsplash.jpg"
pixels = plt.imread(filename)
print("Shape of image/array:", pixels.shape)
plt.imshow(pixels)
plt.show()

# Draw an image with detected objects
def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(rect)

        # Draw the keypoints
        for key, value in result['keypoints'].items():
            dot = plt.Circle(value, radius=20, color='orange')
            ax.add_patch(dot)
    
    # Show the plot
    plt.show()

# Example of using the draw_facebox function
filename = 'test1.jpg'  # Replace with actual filename
pixels = plt.imread(filename)  # Load image from file
detector = mtcnn.MTCNN()  # Initialize MTCNN detector
faces = detector.detect_faces(pixels)  # Detect faces in the image
draw_facebox(filename, faces)  # Display faces on the original image
torch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm
import mtcnn
import matplotlib.pyplot as plt

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define our MTCNN extractor
fast_mtcnn = MTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()

    for filename in tqdm(filenames):
        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):
            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:
                faces = fast_mtcnn(frames)
                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []

                print(
                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )
        v_cap.stop()

# Example filenames for detection
filenames = ['video1.mp4', 'video2.mp4']  # Replace with actual filenames
run_detection(fast_mtcnn, filenames)

# Print version of mtcnn
print(mtcnn._version_)

# Load image from file and display
filename = "glediston-bastos-ZtmmR9D_2tA-unsplash.jpg"
pixels = plt.imread(filename)
print("Shape of image/array:", pixels.shape)
plt.imshow(pixels)
plt.show()

# Draw an image with detected objects
def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
        ax.add_patch(rect)

        # Draw the keypoints
        for key, value in result['keypoints'].items():
            dot = plt.Circle(value, radius=20, color='orange')
            ax.add_patch(dot)
    
    # Show the plot
    plt.show()

# Example of using the draw_facebox function
filename = 'test1.jpg'  # Replace with actual filename
pixels = plt.imread(filename)  # Load image from file
detector = mtcnn.MTCNN()  # Initialize MTCNN detector
faces = detector.detect_faces(pixels)  # Detect faces in the image
draw_facebox(filename, faces)  # Display faces on the original image
