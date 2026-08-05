mod detection;
mod model;
mod recognition;

use std::env;
use std::path::{Path, PathBuf};

use clap::Args;
use image::DynamicImage;
use ort::session::Session;
use ort::value::Tensor;
use thiserror::Error;

use detection::TextBox;

#[derive(Debug, Args)]
pub(crate) struct ImageArgs {
    /// Input image path
    #[arg(value_name = "PNG")]
    pub(crate) image: PathBuf,
}

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("input image is invalid: {0}")]
    Image(#[from] image::ImageError),

    #[error("model is not installed: {0}")]
    ModelNotInstalled(PathBuf),

    #[error("model manifest is invalid: {0}")]
    Manifest(String),

    #[error("model selection failed: {0}")]
    ModelSelection(String),

    #[error("model file is invalid: {0}")]
    ModelFile(String),

    #[error("ONNX Runtime error: {0}")]
    Runtime(String),
}

pub fn run(image_path: PathBuf) -> Result<(), OcrError> {
    let text = recognize(image_path)?;
    println!("{text}");
    Ok(())
}

pub fn recognize(image_path: PathBuf) -> Result<String, OcrError> {
    crate::models::sync_config().map_err(|error| OcrError::ModelSelection(error.to_string()))?;
    let profile = crate::models::selected_profile()
        .map_err(|error| OcrError::ModelSelection(error.to_string()))?;
    let model_dir = crate::models::model_path(&profile);
    if !model_dir.is_dir() {
        return Err(OcrError::ModelNotInstalled(model_dir));
    }
    let (manifest, charset) = model::load(&model_dir)?;
    let det_model = model_dir.join(&manifest.det.model);
    let rec_model = model_dir.join(&manifest.rec.model);
    let image = image::open(image_path)?;

    init_runtime()?;
    let mut detector = load_session(&det_model)?;
    let mut recognizer = load_session(&rec_model)?;
    let boxes = run_detection(&mut detector, &image)?;
    let mut lines = Vec::with_capacity(boxes.len());
    for text_box in boxes {
        let crop = recognition::crop(&image, text_box)?;
        let output = run_recognition(&mut recognizer, recognition::preprocess(crop)?)?;
        lines.push(recognition::decode_ctc(
            &output,
            &charset,
            manifest.blank_index,
            manifest.space_index,
        )?);
    }
    if lines.is_empty() {
        let output = run_recognition(&mut recognizer, recognition::preprocess(image)?)?;
        lines.push(recognition::decode_ctc(
            &output,
            &charset,
            manifest.blank_index,
            manifest.space_index,
        )?);
    }
    Ok(lines.join("\n"))
}

fn init_runtime() -> Result<(), OcrError> {
    let library = env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/lib/libonnxruntime.so"));
    ort::init_from(&library)
        .map_err(|error| {
            OcrError::Runtime(format!("could not load {}: {error}", library.display()))
        })?
        .commit();
    Ok(())
}

fn load_session(model_path: &Path) -> Result<Session, OcrError> {
    Session::builder()
        .map_err(|error| OcrError::Runtime(error.to_string()))?
        .commit_from_file(model_path)
        .map_err(|error| {
            OcrError::Runtime(format!("could not load {}: {error}", model_path.display()))
        })
}

fn run_recognition(session: &mut Session, input: Vec<f32>) -> Result<Vec<f32>, OcrError> {
    let tensor = Tensor::from_array(([1_usize, 3, 48, 320], input))
        .map_err(|error| OcrError::Runtime(format!("could not create input tensor: {error}")))?;
    run_tensor(session, tensor)
}

fn run_detection(session: &mut Session, image: &DynamicImage) -> Result<Vec<TextBox>, OcrError> {
    let prepared = detection::prepare(image)?;
    let geometry = prepared.geometry();
    let tensor = Tensor::from_array((
        [
            1_usize,
            3,
            prepared.padded_height as usize,
            prepared.padded_width as usize,
        ],
        prepared.tensor,
    ))
    .map_err(|error| OcrError::Runtime(format!("could not create input tensor: {error}")))?;
    let (shape, values) = run_tensor_with_shape(session, tensor)?;
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
        return Err(OcrError::Runtime(format!(
            "unsupported detection output shape {shape:?}"
        )));
    }
    let output_height = usize::try_from(shape[2])
        .map_err(|_| OcrError::Runtime(format!("invalid detection height {}", shape[2])))?;
    let output_width = usize::try_from(shape[3])
        .map_err(|_| OcrError::Runtime(format!("invalid detection width {}", shape[3])))?;
    if output_width == 0 || output_height == 0 {
        return Err(OcrError::Runtime(
            "detection output has an empty dimension".to_owned(),
        ));
    }
    if values.len() != output_height.saturating_mul(output_width) {
        return Err(OcrError::Runtime(
            "detection output size does not match shape".to_owned(),
        ));
    }
    Ok(detection::postprocess(
        &values,
        output_width,
        output_height,
        geometry,
    ))
}

fn run_tensor(session: &mut Session, tensor: Tensor<f32>) -> Result<Vec<f32>, OcrError> {
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|error| OcrError::Runtime(format!("inference failed: {error}")))?;
    let (_, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|error| OcrError::Runtime(format!("could not read model output: {error}")))?;
    Ok(values.to_vec())
}

fn run_tensor_with_shape(
    session: &mut Session,
    tensor: Tensor<f32>,
) -> Result<(Vec<i64>, Vec<f32>), OcrError> {
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|error| OcrError::Runtime(format!("inference failed: {error}")))?;
    let (shape, values) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|error| OcrError::Runtime(format!("could not read model output: {error}")))?;
    Ok((shape.to_vec(), values.to_vec()))
}
