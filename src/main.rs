use pyo3::PyObject;

mod model;


fn main() {

    let mut yolo_module = model::DetModule::new();
    yolo_module.load_model("yolov8n.pt");
    let res = yolo_module.infer("/Users/new/Downloads/car.png");
    match res {
        None => {}
        Some(res) => println!("res: {:?}", res)
    };
}
