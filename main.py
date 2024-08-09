from time import time

import cv2
import torch
import os
import sys

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from pygame import mixer


def send_email(object_detected=1):
    """Sends an email notification indicating the number of objects detected; defaults to 1 object."""
    mixer.init()
    mixer.music.load("alert.mp3")
    mixer.music.play()
    print("alarm")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ObjectDetection:
    def __init__(self, capture_index):
        """Initializes an ObjectDetection instance with a given camera index."""
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        model_path = resource_path("best.pt")
        self.model = YOLO(model_path)

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        CONFIDENCE = 0.55  # 0.35
        results = self.model(im0, conf=CONFIDENCE)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (255, 255, 255),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            if cls == 5 or cls == 0:
                class_ids.append(cls)
                self.annotator.box_label(
                    box, label=names[int(cls)], color=colors(int(cls), True)
                )
        return im0, class_ids

    def __call__(self):
        """Run object detection on video frames from a camera stream, plotting and showing the results."""
        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if (5 in class_ids) and (
                0 in class_ids
            ):  # Only send email If not sent before
                if not self.email_sent:
                    send_email(len(class_ids))
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow("YOLOv8 Detection", im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetection(capture_index=0)
    detector()
