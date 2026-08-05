use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use imageproc::point::Point;

use super::OcrError;
use super::detection::TextBox;

const INPUT_HEIGHT: u32 = 48;
const INPUT_WIDTH: u32 = 320;

pub(crate) fn preprocess(image: DynamicImage) -> Result<Vec<f32>, OcrError> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(OcrError::Image(image::ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        )));
    }
    let resized_width =
        ((width as f32 * INPUT_HEIGHT as f32 / height as f32).ceil() as u32).clamp(1, INPUT_WIDTH);
    let resized = image
        .resize_exact(resized_width, INPUT_HEIGHT, FilterType::Lanczos3)
        .to_rgb8();
    let mut tensor = vec![0.0_f32; (3 * INPUT_HEIGHT * INPUT_WIDTH) as usize];
    for y in 0..INPUT_HEIGHT {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x, y).0;
            let bgr = [pixel[2], pixel[1], pixel[0]];
            for (channel, value) in bgr.iter().enumerate() {
                let index = channel * (INPUT_HEIGHT * INPUT_WIDTH) as usize
                    + y as usize * INPUT_WIDTH as usize
                    + x as usize;
                tensor[index] = *value as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(tensor)
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
    use super::decode_ctc;

    #[test]
    fn decodes_ctc() {
        let values = [
            0.0, 0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0,
            0.9, 0.0, 0.0,
        ];
        let text = decode_ctc(&values, &['a', 'b'], 0, 3).unwrap();
        assert_eq!(text, "aba");
    }
}
