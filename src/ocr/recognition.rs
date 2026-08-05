use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use imageproc::point::Point;

use super::OcrError;
use super::detection::TextBox;

const INPUT_HEIGHT: u32 = 48;
const INPUT_WIDTH_MIN: u32 = 320;
const INPUT_WIDTH_MAX: u32 = 8192;
const OVERLAP_WIDTH: u32 = 128;
const CHUNKS_MAX: usize = 64;

pub(crate) struct RecognitionInput {
    pub(crate) tensor: Vec<f32>,
    pub(crate) width: usize,
}

pub(crate) fn preprocess(image: DynamicImage) -> Result<RecognitionInput, OcrError> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(OcrError::Image(image::ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        )));
    }
    let resized_width = (width as f64 * INPUT_HEIGHT as f64 / height as f64).ceil() as u32;
    if resized_width > INPUT_WIDTH_MAX {
        return Err(OcrError::Runtime(format!(
            "recognition input width {resized_width} exceeds maximum {INPUT_WIDTH_MAX}"
        )));
    }
    let input_width = resized_width.max(INPUT_WIDTH_MIN);
    let resized = image
        .resize_exact(resized_width, INPUT_HEIGHT, FilterType::Lanczos3)
        .to_rgb8();
    let mut tensor = vec![0.0_f32; (3 * INPUT_HEIGHT * input_width) as usize];
    for y in 0..INPUT_HEIGHT {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x, y).0;
            let bgr = [pixel[2], pixel[1], pixel[0]];
            for (channel, value) in bgr.iter().enumerate() {
                let index = channel * (INPUT_HEIGHT * input_width) as usize
                    + y as usize * input_width as usize
                    + x as usize;
                tensor[index] = *value as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(RecognitionInput {
        tensor,
        width: input_width as usize,
    })
}

pub(crate) fn crop(image: &DynamicImage, text_box: TextBox) -> Result<DynamicImage, OcrError> {
    let points = text_box.points;
    let width = distance(points[0], points[1]).max(distance(points[2], points[3]));
    let height = distance(points[0], points[3]).max(distance(points[1], points[2]));
    let output_width = width.round().max(1.0) as u32;
    let output_height = height.round().max(1.0) as u32;
    let destination = [
        (0.0, 0.0),
        (output_width as f32, 0.0),
        (output_width as f32, output_height as f32),
        (0.0, output_height as f32),
    ];
    let source = points.map(|point| (point.x, point.y));
    let projection = Projection::from_control_points(source, destination).ok_or_else(|| {
        OcrError::Runtime("could not construct text-box perspective transform".to_owned())
    })?;
    let source = image.to_rgb8();
    let mut output = RgbImage::new(output_width, output_height);
    warp_into(
        &source,
        projection,
        Interpolation::Bicubic,
        Border::Replicate,
        &mut output,
    );
    let output = if output_height as f32 / output_width as f32 >= 1.5 {
        image::imageops::rotate90(&output)
    } else {
        output
    };
    Ok(DynamicImage::ImageRgb8(output))
}

pub(crate) fn crop_line(
    image: &DynamicImage,
    text_boxes: &[TextBox],
) -> Result<DynamicImage, OcrError> {
    if text_boxes.len() == 1 {
        return crop(image, text_boxes[0]);
    }
    let left = text_boxes
        .iter()
        .map(|text_box| text_box.bounds.left)
        .min()
        .unwrap_or(0);
    let right = text_boxes
        .iter()
        .map(|text_box| text_box.bounds.right)
        .max()
        .unwrap_or(left);
    let top = text_boxes
        .iter()
        .map(|text_box| text_box.bounds.top)
        .min()
        .unwrap_or(0);
    let bottom = text_boxes
        .iter()
        .map(|text_box| text_box.bounds.bottom)
        .max()
        .unwrap_or(top);
    let padding = bottom.saturating_sub(top);
    let right = right.saturating_add(padding).min(image.width());
    if right <= left || bottom <= top {
        return Err(OcrError::Runtime(
            "detected text line has invalid bounds".to_owned(),
        ));
    }
    Ok(image.crop_imm(left, top, right - left, bottom - top))
}

pub(crate) fn split(image: DynamicImage) -> Result<Vec<DynamicImage>, OcrError> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(OcrError::Image(image::ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        )));
    }
    let width_max = height.saturating_mul(INPUT_WIDTH_MAX) / INPUT_HEIGHT;
    if width <= width_max.max(1) {
        return Ok(vec![image]);
    }
    let overlap = height
        .saturating_mul(OVERLAP_WIDTH)
        .div_ceil(INPUT_HEIGHT)
        .max(1)
        .min(width_max.saturating_sub(1));
    let step = width_max.saturating_sub(overlap).max(1);
    let count = 1 + width.saturating_sub(width_max).div_ceil(step);
    if count as usize > CHUNKS_MAX {
        return Err(OcrError::Runtime(format!(
            "text line requires {count} recognition chunks; maximum is {CHUNKS_MAX}"
        )));
    }
    let mut chunks = Vec::with_capacity(count as usize);
    let mut left: u32 = 0;
    loop {
        let right = left.saturating_add(width_max).min(width);
        chunks.push(image.crop_imm(left, 0, right - left, height));
        if right == width {
            break;
        }
        left = right - overlap;
    }
    Ok(chunks)
}

fn distance(left: Point<f32>, right: Point<f32>) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

pub(crate) fn decode_ctc(
    values: &[f32],
    charset: &[char],
    blank_index: usize,
    space_index: usize,
) -> Result<String, OcrError> {
    let class_count = charset.len() + 2;
    if !values.len().is_multiple_of(class_count) {
        return Err(OcrError::Runtime(format!(
            "model output length {} is not divisible by class count {class_count}",
            values.len()
        )));
    }
    let mut text = String::new();
    let mut previous = blank_index;
    for timestep in values.chunks_exact(class_count) {
        let (class, _) = timestep
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| OcrError::Runtime("model output is empty".to_owned()))?;
        if class != blank_index && class != previous {
            if class == space_index {
                text.push(' ');
            } else if let Some(character) = charset.get(class.saturating_sub(1)) {
                text.push(*character);
            } else {
                return Err(OcrError::Runtime(format!(
                    "model returned invalid class index {class}"
                )));
            }
        }
        previous = class;
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GenericImageView};

    use super::{decode_ctc, preprocess, split};

    #[test]
    fn decodes_ctc() {
        let values = [
            0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0,
            0.9, 0.0, 0.0,
        ];
        let text = decode_ctc(&values, &['a', 'b'], 0, 3).unwrap();
        assert_eq!(text, "aba");
    }

    #[test]
    fn splits_long_line() {
        let chunks = split(DynamicImage::new_rgb8(20_000, 48)).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].dimensions(), (8192, 48));
        assert_eq!(chunks.last().unwrap().height(), 48);
    }

    #[test]
    fn preserves_long_line() {
        let input = preprocess(DynamicImage::new_rgb8(1200, 48)).unwrap();

        assert_eq!(input.width, 1200);
        assert_eq!(input.tensor.len(), 3 * 48 * 1200);
    }
}
