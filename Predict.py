from ultralytics import YOLO
import numpy as np

model = YOLO(r"D:\Yugam\VS Code\SIH\SIH2\train26\weights\last.pt") #change to path of best.pt

results = model(r"D:\Yugam\VS Code\SIH\output_faces\WhatsApp Image 2024-09-01 at 14.54.22_3b881198.jpg") #change to path of image to test

names_dict = results[0].names

probs = results[0].probs.data.cpu().numpy().tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])
