use pyo3::prelude::*;
use pyo3::types::PyTuple;

static PYTHON_FILE_PATH: &str = "yolov8.py";

pub struct DetModule {
    pub model_module: Option<PyObject>,
    pub func_load_model: Option<Py<PyAny>>,
    pub func_infer: Option<Py<PyAny>>,
}

impl DetModule {
    pub fn new() -> Self {
        let mut det_module = Self {
            model_module: None,
            func_load_model: None,
            func_infer: None,
        };

        let python_file_path = include_str!("yolov8.py");

        Python::with_gil(|py| {
            let det: &PyModule = PyModule::from_code(py, python_file_path, "", "")
                .expect("Load det module error");

            let func_load_model: Py<PyAny> = det.getattr("load_model")
                .expect("Get func 'load_model' error").into();

            let func_infer: Py<PyAny> = det.getattr("infer")
                .expect("Get func 'infer' error").into();

            det_module.func_load_model = Some(func_load_model);
            det_module.func_infer = Some(func_infer);
        });

        det_module
    }

    pub fn load_model(&mut self, model_file: &str) {
        Python::with_gil(|py| {
            let load_model_args = PyTuple::new(py, &[model_file]);
            let model_module = self.func_load_model.call1(py, load_model_args)
                .expect("Load model file error");

            self.model_module = Some(model_module);
        });
    }

    pub fn infer(&self, img_path: &str) -> Option<PyResult<PyObject>> {
        let mut infer_res = None;

        Python::with_gil(|py| {
            match self.model_module.unwrap() {
                None => PyErr::new("The module is uninitialized"),
                Some(model_module) => {
                    let infer_args = PyTuple::new(py, &[model_module, img_path]);
                    infer_res = Option::from(
                        self.func_infer.expect("The module is uninitialized")
                            .call1(py, infer_args));
                }
            }
        });

        infer_res
    }
}




