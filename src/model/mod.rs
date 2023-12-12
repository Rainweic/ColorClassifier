use pyo3::prelude::*;

static PYTHON_FILE_PATH: &str = include_str!("yolov8.py");

pub struct DetModule {
    pub instance: Option<Py<PyAny>>
}

impl DetModule {
    pub fn new() -> Self {

        let mut det_module = Self { instance: None };

        Python::with_gil(|py| {
            let det: &PyModule = PyModule::from_code(py, PYTHON_FILE_PATH, "", "")
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

    pub fn infer(&self, img_path: &str) -> Option<PyObject> {
        let mut infer_res = None;

        Python::with_gil(|py| {
            let args = (img_path,);
            let res = self.instance.as_ref().unwrap()
                .call_method1(py, "infer", args)
                .expect("Infer error, img path: {img_path}");
            // infer_res = Some(res);
        });

        infer_res
    }
}




