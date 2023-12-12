from ultralytics import YOLO


class YoloV8:

    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def infer(self, img_path):
        res = self.model(img_path)
        return res


yolov8 = YoloV8()
