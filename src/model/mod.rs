use pyo3::prelude::*;
use serde::Deserialize;
use serde_json;


#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Box {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct DetResult {
    name: String,
    class: u32,
    confidence: f64,
    bbox: Box
}

pub struct DetModule {
    pub instance: Option<Py<PyAny>>
}

impl DetModule {
    pub fn new() -> Self {

        let mut det_module = Self { instance: None };
        let python_file_path = include_str!("yolov8.py");

        Python::with_gil(|py| {
            let det: &PyModule = PyModule::from_code(py, python_file_path, "yolov8.py", "yolo")
                .expect("Load det module error");
            let instance: Py<PyAny> = det.getattr("yolov8")
                .expect("Get obj yolov8 failed").into();
            det_module.instance = Some(instance);
        });

        det_module
    }

    pub fn load_model(&mut self, model_file: &str) {
        Python::with_gil(|py| {
            let args = (model_file,);
            self.instance.as_ref().unwrap()
                .call_method1(py, "load_model", args)
                .expect("Load model {model_file} failed");
        });
    }

    pub fn infer(&self, img_path: &str) -> Vec<DetResult> {

        let mut infer_res: Vec<DetResult> = vec![];

        Python::with_gil(|py| {
            let args = (img_path,);
            let res_str: Result<String, PyErr> = self.instance.as_ref().unwrap()
                .call_method1(py, "infer", args)
                .expect("Infer error, img path: {img_path}").extract(py);
            match res_str {
                Ok(res_str) => {
                    let res_str = res_str.replace("box", "bbox");
                    infer_res = serde_json::from_str(res_str.as_str()).expect("反序列化错误");
                    println!("{:?}", infer_res);
                },
                Err(err) => {
                    println!("{:?}", err.to_string());
                }
            }
        });

        infer_res
    }
}