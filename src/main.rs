mod model;


fn main() {

    // For test yolov8.py
    let mut yolo_module = model::DetModule::new();
    yolo_module.load_model("yolov8n.pt");
    yolo_module.infer("/home/cu/ColorClassifier/car.jpg");

}
