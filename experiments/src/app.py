from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")  # o un modelo más potente

cap = cv2.VideoCapture("/Volumes/ENDERCHEST/bSide/Proyectos/COMPUTER-VISION-DEMO/data/bronze/parking_3.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)      # detección simple
    annotated_frame = results[0].plot()
    cv2.imshow("Detect", annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
