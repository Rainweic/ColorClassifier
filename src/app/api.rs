use rocket::fs::TempFile;
use rocket::{get, post};

#[post("/upload", format="multipart", data="<img>")]
pub async fn api_color_extraction(mut img: TempFile<'_>) -> std::io::Result<()> {
    img.persist_to("./save.jpg").await
}

#[get("/status")]
pub async fn api_status() -> &'static str {
    "OK"
}