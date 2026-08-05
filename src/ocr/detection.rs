use clipper2_rust::{EndType, JoinType, inflate_paths_d};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, GrayImage, ImageError, Luma};
use imageproc::contours::{BorderType, find_contours};
use imageproc::geometry::{approximate_polygon_dp, arc_length, min_area_rect};
use imageproc::point::Point;

use super::OcrError;

const MAX_SIDE: u32 = 960;
const STRIDE: u32 = 32;
const THRESHOLD: f32 = 0.2;
const BOX_THRESHOLD: f32 = 0.45;
const MAX_CANDIDATES: usize = 3000;
const UNCLIP_RATIO: f64 = 1.4;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug)]
pub(crate) struct DetectionInput {
    pub(crate) tensor: Vec<f32>,
    pub(crate) resized_width: u32,
    pub(crate) resized_height: u32,
    pub(crate) padded_width: u32,
    pub(crate) padded_height: u32,
    pub(crate) original_width: u32,
    pub(crate) original_height: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rect {
    pub(crate) left: u32,
    pub(crate) top: u32,
    pub(crate) bottom: u32,
}

impl Rect {
    pub(crate) fn height(self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TextBox {
    pub(crate) points: [Point<f32>; 4],
    pub(crate) bounds: Rect,
}

#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    output_width: usize,
    output_height: usize,
    resized_width: u32,
    resized_height: u32,
    padded_width: u32,
    padded_height: u32,
    original_width: u32,
    original_height: u32,
}

impl DetectionInput {
    pub(crate) fn geometry(&self) -> Geometry {
        Geometry {
            resized_width: self.resized_width,
            resized_height: self.resized_height,
            padded_width: self.padded_width,
            padded_height: self.padded_height,
            original_width: self.original_width,
            original_height: self.original_height,
            output_width: self.padded_width as usize,
            output_height: self.padded_height as usize,
        }
    }
}

pub(crate) fn prepare(image: &DynamicImage) -> Result<DetectionInput, OcrError> {
    let (original_width, original_height) = image.dimensions();
    if original_width == 0 || original_height == 0 {
        return Err(OcrError::Image(ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        )));
    }
    let scale = (MAX_SIDE as f32 / original_width.max(original_height) as f32).min(1.0);
    let resized_width = ((original_width as f32 * scale).round() as u32).max(1);
    let resized_height = ((original_height as f32 * scale).round() as u32).max(1);
    let padded_width = round_up(resized_width, STRIDE);
    let padded_height = round_up(resized_height, STRIDE);
    let resized = image
        .resize_exact(resized_width, resized_height, FilterType::Lanczos3)
        .to_rgb8();
    let plane_size = (padded_width * padded_height) as usize;
    let mut tensor = vec![0.0_f32; plane_size * 3];
    for channel in 0..3 {
        let padding = (0.0 - MEAN[channel]) / STD[channel];
        tensor[channel * plane_size..(channel + 1) * plane_size].fill(padding);
    }
    for y in 0..resized_height {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x, y).0;
            let bgr = [pixel[2], pixel[1], pixel[0]];
            for (channel, value) in bgr.iter().enumerate() {
                let index = channel * plane_size + y as usize * padded_width as usize + x as usize;
                tensor[index] = (*value as f32 / 255.0 - MEAN[channel]) / STD[channel];
            }
        }
    }
    Ok(DetectionInput {
        tensor,
        resized_width,
        resized_height,
        padded_width,
        padded_height,
        original_width,
        original_height,
    })
}

pub(crate) fn postprocess(
    values: &[f32],
    width: usize,
    height: usize,
    mut geometry: Geometry,
) -> Vec<TextBox> {
    let bitmap = GrayImage::from_fn(width as u32, height as u32, |x, y| {
        if values[y as usize * width + x as usize] > THRESHOLD {
            Luma([255])
        } else {
            Luma([0])
        }
    });
    geometry.output_width = width;
    geometry.output_height = height;
    let mut boxes = Vec::new();
    for contour in find_contours::<i32>(&bitmap)
        .into_iter()
        .take(MAX_CANDIDATES)
    {
        if contour.border_type == BorderType::Hole || contour.points.len() < 3 {
            continue;
        }
        let epsilon = 0.002 * arc_length(&contour.points, true);
        if epsilon <= 0.0 {
            continue;
        }
        let points = approximate_polygon_dp(&contour.points, epsilon, true);
        if points.len() < 4 {
            continue;
        }
        let points = points
            .iter()
            .map(|point| Point::new(point.x as f32, point.y as f32))
            .collect::<Vec<_>>();
        let mini_box = min_area_rect(&points);
        let side = min_side(&mini_box);
        if side < 3.0 {
            continue;
        }
        let score = polygon_score(values, width, height, &mini_box);
        if score < BOX_THRESHOLD {
            continue;
        }
        let Some(expanded) = unclip(&mini_box) else {
            continue;
        };
        let expanded_box = min_area_rect(&expanded);
        if min_side(&expanded_box) < 5.0 {
            continue;
        }
        boxes.push(map_quad(expanded_box, geometry));
    }
    boxes.sort_by(|left, right| {
        let tolerance = left.bounds.height().min(right.bounds.height()) / 2;
        if left.bounds.top.abs_diff(right.bounds.top) <= tolerance {
            left.bounds.left.cmp(&right.bounds.left)
        } else {
            left.bounds.top.cmp(&right.bounds.top)
        }
    });
    boxes
}

fn round_up(value: u32, multiple: u32) -> u32 {
    value.div_ceil(multiple) * multiple
}

fn min_side(points: &[Point<f32>; 4]) -> f32 {
    points
        .windows(2)
        .map(|edge| distance(edge[0], edge[1]))
        .fold(f32::MAX, f32::min)
        .min(distance(points[0], points[3]))
}

fn map_quad(points: [Point<f32>; 4], geometry: Geometry) -> TextBox {
    let scale_x = geometry.original_width as f32 * geometry.padded_width as f32
        / (geometry.resized_width as f32 * geometry.output_width as f32);
    let scale_y = geometry.original_height as f32 * geometry.padded_height as f32
        / (geometry.resized_height as f32 * geometry.output_height as f32);
    let points = points.map(|point| {
        Point::new(
            (point.x * scale_x).clamp(0.0, geometry.original_width as f32),
            (point.y * scale_y).clamp(0.0, geometry.original_height as f32),
        )
    });
    let left = points
        .iter()
        .map(|point| point.x.round())
        .fold(geometry.original_width as f32, f32::min) as u32;
    let top = points
        .iter()
        .map(|point| point.y.round())
        .fold(geometry.original_height as f32, f32::min) as u32;
    let bottom = points
        .iter()
        .map(|point| point.y.round())
        .fold(0.0, f32::max) as u32;
    TextBox {
        points,
        bounds: Rect {
            left,
            top,
            bottom: bottom
                .max(top.saturating_add(1))
                .min(geometry.original_height),
        },
    }
}

fn distance(left: Point<f32>, right: Point<f32>) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn polygon_score(values: &[f32], width: usize, height: usize, polygon: &[Point<f32>]) -> f32 {
    let xmin = polygon
        .iter()
        .map(|point| point.x.floor() as isize)
        .min()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as isize) as usize;
    let xmax = polygon
        .iter()
        .map(|point| point.x.ceil() as isize)
        .max()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as isize) as usize;
    let ymin = polygon
        .iter()
        .map(|point| point.y.floor() as isize)
        .min()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as isize) as usize;
    let ymax = polygon
        .iter()
        .map(|point| point.y.ceil() as isize)
        .max()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as isize) as usize;
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, polygon) {
                sum += values[y * width + x];
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn point_in_polygon(x: f32, y: f32, polygon: &[Point<f32>]) -> bool {
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        let crosses = (current.y > y) != (previous.y > y);
        if crosses {
            let intersection =
                (previous.x - current.x) * (y - current.y) / (previous.y - current.y) + current.x;
            if x < intersection {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn unclip(points: &[Point<f32>; 4]) -> Option<Vec<Point<f32>>> {
    let area = polygon_area(points);
    let length = points
        .iter()
        .enumerate()
        .map(|(index, point)| distance(*point, points[(index + 1) % points.len()]))
        .map(f64::from)
        .sum::<f64>();
    if area <= 0.0 || length <= f64::EPSILON {
        return None;
    }
    let path = vec![
        points
            .iter()
            .map(|point| clipper2_rust::Point::new(point.x as f64, point.y as f64))
            .collect(),
    ];
    inflate_paths_d(
        &path,
        area * UNCLIP_RATIO / length,
        JoinType::Round,
        EndType::Polygon,
        2.0,
        3,
        0.25,
    )
    .first()
    .map(|path| {
        path.iter()
            .map(|point| Point::new(point.x as f32, point.y as f32))
            .collect()
    })
}

fn polygon_area(points: &[Point<f32>; 4]) -> f64 {
    points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let next = points[(index + 1) % points.len()];
            f64::from(point.x) * f64::from(next.y) - f64::from(point.y) * f64::from(next.x)
        })
        .sum::<f64>()
        .abs()
        * 0.5
}

#[cfg(test)]
mod tests {
    use imageproc::point::Point;

    use super::{Geometry, map_quad, polygon_score, unclip};

    #[test]
    fn scores_polygon() {
        let values = vec![1.0; 16];
        let polygon = [
            Point::new(1.0, 1.0),
            Point::new(3.0, 1.0),
            Point::new(3.0, 3.0),
            Point::new(1.0, 3.0),
        ];
        assert_eq!(polygon_score(&values, 4, 4, &polygon), 1.0);
    }

    #[test]
    fn unclip_box() {
        let box_points = [
            Point::new(10.0, 10.0),
            Point::new(30.0, 10.0),
            Point::new(30.0, 20.0),
            Point::new(10.0, 20.0),
        ];
        let expanded = unclip(&box_points).expect("box should expand");
        let min_x = expanded
            .iter()
            .map(|point| point.x)
            .fold(f32::MAX, f32::min);
        let max_x = expanded
            .iter()
            .map(|point| point.x)
            .fold(f32::MIN, f32::max);
        assert!(min_x < 10.0);
        assert!(max_x > 30.0);
    }

    #[test]
    fn detects_region() {
        let mut values = vec![0.0; 64];
        for y in 2..6 {
            for x in 2..6 {
                values[y * 8 + x] = 0.9;
            }
        }
        let geometry = super::Geometry {
            output_width: 8,
            output_height: 8,
            resized_width: 8,
            resized_height: 8,
            padded_width: 8,
            padded_height: 8,
            original_width: 80,
            original_height: 80,
        };
        let boxes = super::postprocess(&values, 8, 8, geometry);
        assert_eq!(boxes.len(), 1);
        assert!(boxes[0].bounds.left < 80);
        assert!(boxes[0].bounds.bottom > boxes[0].bounds.top);
    }

    #[test]
    fn maps_padding() {
        let geometry = Geometry {
            output_width: 128,
            output_height: 32,
            resized_width: 100,
            resized_height: 20,
            padded_width: 128,
            padded_height: 32,
            original_width: 100,
            original_height: 20,
        };
        let points = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 20.0),
            Point::new(0.0, 20.0),
        ];

        let text_box = map_quad(points, geometry);

        assert_eq!(text_box.points[1].x, 100.0);
        assert_eq!(text_box.bounds.bottom, 20);
    }
}
