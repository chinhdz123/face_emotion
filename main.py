import cv2
import numpy as np
import copy
from ultralytics import YOLO
from predict_ncnn import SqueezeNet

model = SqueezeNet()
model_yolo = YOLO("models/yolov8n-face.pt")

labels = ['happy','neutral']

def detect_faces_by_yolo(image, is_save = True):
    
    results = model_yolo.predict(image, conf = 0.5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            if w/h > 1.2 or w/h < 0.7 or w < 50 or h < 50:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                continue
            if w/h > 1.2 or w/h < 0.3 or w < 50 or h < 50:
                continue
            img_crop = image[y1:y2,x1:x2]
            img_save = copy.deepcopy(img_crop)
            predictions = model(img_crop)
            score= predictions[np.argmax(predictions)]
            label = labels[np.argmax(predictions)] 
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if score > 0.9:
                cv2.putText(image, label, (x1,y1-20), cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255))
    return image
            

        
def preprocess_data(image):
    img_input = cv2.resize(image, (66, 66))
    img_input = img_input / 255.0
    img_input = np.array([img_input])
    print(img_input.shape)
    return img_input

def main():
    name = "atv"
    cap = cv2.VideoCapture(f"videos/vlc-output.ts")
    # cap = cv2.VideoCapture(f"video/{name}.mp4")
    frame_width = int(cap.get(3)/2)
    frame_height = int(cap.get(4)/2)
    size = (frame_width, frame_height)
    save_path = f"outputs/{name}.mp4"
    result = cv2.VideoWriter(save_path, 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        10, size)
    i = 0
    while True:
        success, img = cap.read() 
        if success:
            i+=1
            is_save = False
            if i%1 == 0:
                # is_save = True
                img = cv2.resize(img, (0,0), fx =0.5, fy =0.5)

                img = detect_faces_by_yolo(img, is_save)
                result.write(img)
                cv2.imshow("a",img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        else:
            break
if __name__ == "__main__":
    main()
