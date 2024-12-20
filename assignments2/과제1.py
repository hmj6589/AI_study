from ultralytics import YOLO

# 과제1 : 이미지 1에 대해 predict
# (yolov8m으로 confidence가 0.2, 0.5, 0.8 일 때 수행)
model = YOLO("yolov8m.pt")   # yolo8m
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8