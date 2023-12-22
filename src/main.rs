use color_classifier::model::DetModule;
use color_classifier::core::rec_color;


fn main() {
    let mut yolo = DetModule::new();
    yolo.load_model("yolov8n.pt");
    rec_color::det_color(yolo, "/home/cu/ColorClassifier/car.jpg");
}

#[cfg(test)]
mod test {

    use super::*;
    use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};

    #[test]
    fn test_segmentation() {
        let img_roi = imread("img/test.png", IMREAD_COLOR).expect("获取图像错误");
        let img_roi_vec = vec![img_roi];
        rec_color::segmentation(img_roi_vec);
    }

}
