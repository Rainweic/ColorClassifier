from ultralytics import YOLO


def load_model(model_path):
    return YOLO(model_path)


def infer(model, img_path):
    res = model(img_path)
    return res
