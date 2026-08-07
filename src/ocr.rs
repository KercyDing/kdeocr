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
    let model_dir = crate::models::model_path(&profile)
        .map_err(|error| OcrError::ModelSelection(error.to_string()))?;
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
    let mut lines = Vec::new();
    for line in detection::group_lines(boxes) {
        let prefix = if let Some(text_box) = detection::separated_prefix(&line) {
            let crop = recognition::crop(&image, text_box)?;
            Some(recognize_crop(
                &mut recognizer,
                crop,
                &charset,
                manifest.blank_index,
                manifest.space_index,
            )?)
        } else {
            None
        };
        let crop = recognition::crop_line(&image, &line)?;
        let text = recognize_crop(
            &mut recognizer,
            crop,
            &charset,
            manifest.blank_index,
            manifest.space_index,
        )?;
        let text = match prefix {
            Some(prefix) => preserve_gap(text, &prefix),
            None => text,
        };
        if !text.is_empty() {
            lines.push(text);
        }
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

fn recognize_crop(
    recognizer: &mut Session,
    crop: DynamicImage,
    charset: &[char],
    blank_index: usize,
    space_index: usize,
) -> Result<String, OcrError> {
    let mut text = String::new();
    for chunk in recognition::split(crop)? {
        let output = run_recognition(recognizer, recognition::preprocess(chunk)?)?;
        let chunk = recognition::decode_ctc(&output, charset, blank_index, space_index)?;
        let overlap = text_overlap(&text, &chunk);
        text.extend(chunk.chars().skip(overlap));
    }
    Ok(text)
}

fn text_overlap(left: &str, right: &str) -> usize {
    let left: Vec<_> = left.chars().collect();
    let right: Vec<_> = right.chars().collect();
    for count in (1..=left.len().min(right.len())).rev() {
        if left[left.len() - count..]
            .iter()
            .zip(&right[..count])
            .all(|(left, right)| left == right || left.eq_ignore_ascii_case(right))
        {
            return count;
        }
    }
    0
}

fn preserve_gap(text: String, prefix: &str) -> String {
    if let Some(suffix) = text.strip_prefix(prefix) {
        if suffix.is_empty() || suffix.starts_with(char::is_whitespace) {
            return text;
        }
        return format!("{prefix} {suffix}");
    }
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return text;
    };
    let suffix = chars.as_str();
    if first.is_alphanumeric() || suffix.is_empty() || suffix.starts_with(char::is_whitespace) {
        return text;
    }
    format!("{first} {suffix}")
}

pub(crate) fn runtime_library() -> PathBuf {
    runtime_library_from(env::var_os("ORT_DYLIB_PATH"))
}

fn runtime_library_from(configured: Option<std::ffi::OsString>) -> PathBuf {
    configured
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("libonnxruntime.so"))
}

pub(crate) fn init_runtime() -> Result<(), OcrError> {
    let library = runtime_library();
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

fn run_recognition(
    session: &mut Session,
    input: recognition::RecognitionInput,
) -> Result<Vec<f32>, OcrError> {
    let tensor = Tensor::from_array(([1_usize, 3, 48, input.width], input.tensor))
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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use super::{preserve_gap, runtime_library_from, text_overlap};

    #[test]
    fn overlaps_case() {
        assert_eq!(text_overlap("fe", "Features"), 2);
    }

    #[test]
    fn preserves_gap() {
        assert_eq!(
            preserve_gap("13[dependencies]".to_owned(), "13"),
            "13 [dependencies]"
        );
        assert_eq!(preserve_gap("● text".to_owned(), "●"), "● text");
        assert_eq!(preserve_gap("●text".to_owned(), "Q"), "● text");
    }

    #[test]
    fn selects_runtime_library() {
        assert_eq!(
            runtime_library_from(None),
            PathBuf::from("libonnxruntime.so")
        );
        assert_eq!(
            runtime_library_from(Some(OsString::from("/custom/libonnxruntime.so"))),
            PathBuf::from("/custom/libonnxruntime.so")
        );
    }
}
