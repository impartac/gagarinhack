from ultralytics import YOLO
import torch
import os
# Load the model.
model = YOLO('yolov8n-cls.pt')
 
 #"C:\\Users\\32233\\PycharmProjects\\temp\\datasets",
def train():
    torch.multiprocessing.freeze_support()
    results = model.train(
            data="C:\\Users\\32233\\PycharmProjects\\temp\\document_classification-1",
            imgsz=640,
            epochs=100,
            batch=8,
            device=0)

if __name__ == '__main__':
    train()