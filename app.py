import cv2
import numpy as np
import os

weights_path = r"D:\ML Projects\naveen\Object-Detection\yolo\yolo.weights"
config_path = r"D:\ML Projects\naveen\Object-Detection\yolo\yolo.cfg"
names_path = r"D:\ML Projects\naveen\Object-Detection\yolo\coco.names"

if not os.path.isfile(weights_path):
    print("Weights file not found:", weights_path)
if not os.path.isfile(config_path):
    print("Config file not found:", config_path)
if not os.path.isfile(names_path):
    print("Names file not found:", names_path)

yolo_net = cv2.dnn.readNet(weights_path, config_path)

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame = detect_objects(frame)

        cv2.imshow("Object Detection", detected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
