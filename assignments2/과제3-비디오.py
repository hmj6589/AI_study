from ultralytics import YOLO

# 동영상
# model = YOLO("yolov8m.pt")   # yolo8m
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8

# model = YOLO("yolov8n.pt")   # yolo8n
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8

# model = YOLO("yolov8n-seg.pt")   # yolo8n-seg
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8

model = YOLO("yolov8m-seg.pt")   # yolo8m-seg
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.2)   # confidence=0.2
# model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.5)   # confidence=0.5
model.predict("./dataset/slow_traffic_small.mp4", show=True, save=True, imgsz=320, conf=0.8)   # confidence=0.8
