use rocket::{launch, routes};
// use color_classifier::model::DetModule;
use color_classifier::core::rec_color;
use color_classifier::app::api::{api_color_extraction, api_status};

#[launch]
fn start() -> _ {
    rocket::build().mount("/api", routes![api_color_extraction, api_status])
}

#[cfg(test)]
mod test {

    use super::*;
    use opencv::core::Mat;
    use opencv::core::Vector;
    use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};

    fn save_img(img_vec: Vec<Mat>, save_suffix: &str) {
        for (i, img) in img_vec.iter().enumerate() {
            let params = Vector::new();
            let filename = format!("{}_{}.jpg", save_suffix, i);
            imwrite(filename.as_str(), &img, &params).expect("保存图像失败");
        }
    }

    #[test]
    fn test_segmentation() {
        let img_roi = imread("img/test.png", IMREAD_COLOR).expect("获取图像错误");
        let img_roi_vec = vec![img_roi];
        let seg_img = rec_color::segmentation(img_roi_vec);
        save_img(seg_img, "seg");
    }

    #[test]
    fn test_cal_mean_value_of_hsv() {
        let img_roi = imread("img/seg.jpg", IMREAD_COLOR).expect("获取图像错误");
        let img_roi_vec = vec![img_roi];
        rec_color::cal_mean_value_of_hsv(img_roi_vec);
    }

}
