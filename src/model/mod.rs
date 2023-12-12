use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};

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
            let model_module = self.func_load_model.as_ref().unwrap()
                .call1(py, load_model_args)
                .expect("Load model file error");

            self.model_module = Some(model_module);
        });
    }

    pub fn infer(&self, img_path: &str) -> Option<PyObject> {
        let mut infer_res = None;

        Python::with_gil(|py| {
            let infer_args = [(self.model_module.as_ref().unwrap(), img_path)]
                .into_py_dict(py);
            let res = self.func_infer.as_ref().unwrap()
                .call(py, (), Some(infer_args))
                .expect("Infer error");
            infer_res = Some(res);
        });

        infer_res
    }
}




