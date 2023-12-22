use opencv::core::{Mat, Rect, Vector};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, COLOR_BGR2GRAY, threshold, THRESH_BINARY, THRESH_OTSU};

use crate::model::{DetModule, DetResult};


/// Step1 提取ROI
/// Step2 做区域分割
/// Step3 做颜色提取
#[allow(dead_code)]
pub fn det_color(yolo: DetModule, img_path: &str) {

    let det_res_vec: Vec<DetResult> = yolo.infer(img_path);
    let img = imread(img_path, IMREAD_COLOR);

    match img {
        Ok(img) => {

            // 提取ROI图像
            let img_roi_vec = extract_roi(det_res_vec, img);

            // Test
            // for (i, _img) in img_roi_vec.iter().enumerate() {
            //     let img_name = format!("img_{:?}.jpg", i);
            //     let params = Vector::new();
            //     imwrite(&img_name, _img, &params).expect("写入图像失败");
            // }

            // 对ROI区域做边缘检测
            let binnary_img_vec = segmentation(img_roi_vec);

            // 颜色提取
        },
        Err(e) => {
            println!("读取图片错误: {:?}", e.to_string());
        }
    }
}

/// 根据Yolo的识别结果，从原图种截取ROI部分
fn extract_roi(det_res_vec: Vec<DetResult>, img: Mat) -> Vec<Mat> {
    let mut ret: Vec<Mat> = vec![];
    for det_res in det_res_vec.iter() {
        let x = det_res.bbox.x1 as i32;
        let y = det_res.bbox.y1 as i32;
        let w = (det_res.bbox.x2 as i32 - x) as i32;
        let h = (det_res.bbox.y2 as i32 - y) as i32;
        // TOOD 后面加超出范围的处理
        let roi_area = Rect::new(x, y, w, h);
        let img_roi = Mat::roi(&img, roi_area).expect("截取ROI区域错误");
        ret.push(img_roi);
    }
    ret
}

/// 对提取出来的ROI区域做分割
pub fn segmentation(img_roi_vec: Vec<Mat>) -> Vec<Mat> {

    let mut ret: Vec<Mat> = vec![];

    for img in img_roi_vec.iter() {

        // 将图像转灰度
        let mut gray_img: Mat = Mat::default();
        cvt_color(img, &mut gray_img, COLOR_BGR2GRAY, 0).expect("BRG2GRAY失败");

        // TEST
        // let params = Vector::new();
        // imwrite("gray_img.jpg", &gray_img, &params);

        // Otsu二值化
        let mut binary_img = Mat::default();
        threshold(
            &gray_img, 
            &mut binary_img, 
            0.0, 
            255.0, 
            THRESH_BINARY + THRESH_OTSU
        ).expect("二值化失败");

        // TEST
        // let params = Vector::new();
        // imwrite("binary_img.jpg", &binary_img, &params);

        ret.push(binary_img);
    }

    ret
}
