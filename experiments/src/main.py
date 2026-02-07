import cv2
from ultralytics import solutions

# Video capture
cap = cv2.VideoCapture("/Volumes/ENDERCHEST/bSide/Proyectos/COMPUTER-VISION-DEMO/data/bronze/parking_2.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking management.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize parking management object
parkingmanager = solutions.ParkingManagement(
    model="yolo26n.pt",  # path to model file
    json_file="bounding_boxes-2.json",  # path to parking annotations file
    classes=[0, 1, 2, 5, 7],

    show=True
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    results = parkingmanager(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows