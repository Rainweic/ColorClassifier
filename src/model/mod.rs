
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub fn load_model(model_path: String) {

    let py_code = include_str!("./yolov8.py");

    Python::with_gil(|py| {
        let load_model = PyModule::from_code(
            py,
            py_code,
            "yolov8.py",
            "YoloV8"
        ).expect("导入yolov8.py文件错误");
    })

}


