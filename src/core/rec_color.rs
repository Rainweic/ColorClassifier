use opencv::core::{Mat, Rect, Point, Vec3b};
use opencv::hub_prelude::MatTraitConst;
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, COLOR_BGR2GRAY, threshold, THRESH_BINARY, THRESH_OTSU, COLOR_BGR2HSV};

use crate::model::{DetModule, DetResult};


/// Step1 提取ROI
/// Step2 做区域分割
/// Step3 做颜色提取
#[allow(dead_code)]
pub fn color_extraction(yolo: DetModule, img_path: &str) {

    let det_res_vec: Vec<DetResult> = yolo.infer(img_path);
    let img = imread(img_path, IMREAD_COLOR);

    match img {
        Ok(img) => {

            // 提取ROI图像
            let img_roi_vec = extract_roi(det_res_vec, img);

            // 对ROI区域做边缘检测
            let img_seg_vec = segmentation(img_roi_vec);

            // 颜色提取
            cal_mean_value_of_hsv(img_seg_vec);
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

        // Otsu二值化
        let mut mask = Mat::default();
        threshold(
            &gray_img, 
            &mut mask, 
            0.0, 
            255.0, 
            THRESH_BINARY + THRESH_OTSU
        ).expect("二值化失败");

        // 利用二值化图像当作mask 从原图中提取区域图像
        let mut seg_img = Mat::default();
        img.copy_to_masked(&mut seg_img, &mask).expect("copy_to_masked失败");

        ret.push(seg_img);
    }

    ret
}

/// 计算HSV均值
pub fn cal_mean_value_of_hsv(binnary_img_vec: Vec<Mat>) -> Vec<(i32, i32, i32)> {

    let mut mean_hsv_vec = vec![];

    for img in binnary_img_vec.iter() {

        // BGR 2 HSV
        let mut img_hsv: Mat = Mat::default();
        cvt_color(img, &mut img_hsv, COLOR_BGR2HSV, 0)
            .expect("BGR2HSV失败");

        // 统计有效像素点均值
        let unvalid_hsv_value = [0, 0, 0];
        let mut n_valid_pt: i32 = 0;
        let mut h_value_sum: i32 = 0;
        let mut s_value_sum: i32 = 0;
        let mut v_value_sum: i32 = 0;

        // 遍历像素点
        let row_number = img_hsv.rows();
        let col_number = img_hsv.cols();

        for i in 0..row_number {
            for j in 0..col_number {
                let pt = Point::new(j, i);
                let hsv_values = img_hsv.at_pt::<Vec3b>(pt).unwrap();
                
                match hsv_values.cmp(&unvalid_hsv_value) {
                    std::cmp::Ordering::Equal => {},
                    _ => {
                        n_valid_pt += 1;
                        h_value_sum += hsv_values[0] as i32;
                        s_value_sum += hsv_values[1] as i32;
                        v_value_sum += hsv_values[2] as i32;
                    }
                }
                
            }
        }

        let mut mean_hsv = (0, 0, 0);
        if n_valid_pt != 0 {
            mean_hsv = (h_value_sum / n_valid_pt, 
                        s_value_sum / n_valid_pt, 
                        v_value_sum / n_valid_pt)
        }

        mean_hsv_vec.push(mean_hsv);
        
    }

    mean_hsv_vec
}