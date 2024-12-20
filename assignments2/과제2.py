from ultralytics import YOLO

# 과제2 : model 변경하며 과제1 수행
# (yolov8n, yolov8n-seg, yolov8m-seg)
# model = YOLO("yolov8n.pt")   # yolo8n
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8

# model = YOLO("yolov8n-seg.pt")   # yolo8n-seg
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8

model = YOLO("yolov8m-seg.pt")   # yolo8m-seg
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
model.predict("./dataset/img1.png", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8
