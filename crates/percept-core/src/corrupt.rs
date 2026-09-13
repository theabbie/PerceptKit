//! Corruption generators — the "labeler is a program" primitives.
//!
//! These produce (corrupted_data, dense_labels, severity) triples for
//! self-supervised training: the corruption process itself is the
//! annotation, so supervision is free and infinite.
//!
//! Two families:
//! - **Layout**: boxes (x, y, w, h) on a page — overlap, overflow,
//!   dead-space, squeeze. Returns per-element labels so the dense head
//!   can learn *which* element is broken, not just that the page is bad.
//! - **Image**: grayscale (for colorization), region masking (for
//!   inpainting), noise. Returns pixel masks for dense supervision.

use crate::rng::Rng;

/// A layout element: [x, y, w, h] in page units.
pub type Box4 = [f32; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayoutCorruption {
    /// Move one box so it overlaps another (colliding elements).
    Overlap,
    /// Shift a box past the right page edge (overflow).
    OverflowRight,
    /// Shift a box past the bottom page edge (overflow).
    OverflowBottom,
    /// Push trailing boxes down, leaving a dead gap (wasted space).
    DeadSpace,
    /// Compress vertical spacing until boxes nearly touch (cramped).
    Squeeze,
}

pub struct CorruptedLayout {
    pub boxes: Vec<Box4>,
    /// Per-element label: 0.0 = intact, 1.0 = this element is broken.
    pub labels: Vec<f32>,
    /// The severity that was applied, in [0, 1].
    pub severity: f32,
}

/// Intersection-over-union of two boxes.
pub fn iou(a: &Box4, b: &Box4) -> f32 {
    let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x2, y2) = ((a[0] + a[2]).min(b[0] + b[2]), (a[1] + a[3]).min(b[1] + b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Apply one corruption kind at a given severity (0..1) to a layout.
/// `page_w`/`page_h` bound the page. Boxes are not required to be sorted.
pub fn corrupt_layout(
    boxes: &[Box4],
    kind: LayoutCorruption,
    severity: f32,
    page_w: f32,
    page_h: f32,
    rng: &mut Rng,
) -> Result<CorruptedLayout, String> {
    if boxes.is_empty() {
        return Err("empty layout".into());
    }
    if !(0.0..=1.0).contains(&severity) {
        return Err(format!("severity {severity} outside [0,1]"));
    }
    if page_w <= 0.0 || page_h <= 0.0 {
        return Err("page dimensions must be positive".into());
    }
    let mut out = boxes.to_vec();
    let mut labels = vec![0.0f32; boxes.len()];

    match kind {
        LayoutCorruption::Overlap => {
            if boxes.len() < 2 {
                return Err("overlap needs at least 2 boxes".into());
            }
            let i = rng.next_usize(boxes.len());
            let mut j = rng.next_usize(boxes.len());
            while j == i {
                j = rng.next_usize(boxes.len());
            }
            // Move box i toward box j by `severity` fraction of the gap
            // plus enough to guarantee actual overlap at severity 1.
            let target = out[j];
            let src = out[i];
            let dx = target[0] - (src[0] + src[2]); // gap to j's left edge
            let dy = target[1] - (src[1] + src[3]);
            // overlap amount: severity 0.5 -> touch, 1.0 -> half-overlap
            let shift_x = dx + severity * src[2].min(target[2]) * 0.5;
            let shift_y = dy + severity * src[3].min(target[3]) * 0.5;
            out[i][0] += shift_x;
            out[i][1] += shift_y;
            labels[i] = 1.0;
            labels[j] = 1.0;
        }
        LayoutCorruption::OverflowRight => {
            let i = rng.next_usize(boxes.len());
            let b = out[i];
            // push so that the box's right edge exceeds page_w by
            // severity * box width
            let excess = severity * b[2];
            out[i][0] = page_w - b[2] + excess;
            labels[i] = 1.0;
        }
        LayoutCorruption::OverflowBottom => {
            let i = rng.next_usize(boxes.len());
            let b = out[i];
            let excess = severity * b[3];
            out[i][1] = page_h - b[3] + excess;
            labels[i] = 1.0;
        }
        LayoutCorruption::DeadSpace => {
            if boxes.len() < 2 {
                return Err("dead space needs at least 2 boxes".into());
            }
            // sort indices by y so "trailing" is well-defined
            let mut idx: Vec<usize> = (0..boxes.len()).collect();
            idx.sort_by(|&a, &b| boxes[a][1].partial_cmp(&boxes[b][1]).unwrap());
            let cut = 1 + rng.next_usize(boxes.len() - 1);
            let gap = severity * page_h * 0.5;
            for &k in &idx[cut..] {
                out[k][1] += gap;
                labels[k] = 1.0;
            }
            labels[idx[cut - 1]] = 1.0; // the box before the gap is "broken" too
        }
        LayoutCorruption::Squeeze => {
            if boxes.len() < 2 {
                return Err("squeeze needs at least 2 boxes".into());
            }
            // Compress all y positions toward the top box by severity.
            let min_y = out.iter().map(|b| b[1]).fold(f32::INFINITY, f32::min);
            for (k, b) in out.iter_mut().enumerate() {
                let gap = b[1] - min_y;
                b[1] = min_y + gap * (1.0 - severity * 0.9);
                labels[k] = if gap > 0.0 { 1.0 } else { labels[k] };
            }
        }
    }

    Ok(CorruptedLayout { boxes: out, labels, severity })
}

// ---------------------------------------------------------------------------
// Image corruptions
// ---------------------------------------------------------------------------

/// Rec. 601 luminance grayscale of an RGB image.
/// Input: (w*h*3) channel-interleaved. Output: (w*h) single channel.
pub fn to_grayscale(img: &[f32], w: usize, h: usize) -> Result<Vec<f32>, String> {
    let expected = w * h * 3;
    if img.len() != expected {
        return Err(format!("rgb len {} != {}", img.len(), expected));
    }
    Ok((0..w * h)
        .map(|i| {
            let r = img[i * 3];
            let g = img[i * 3 + 1];
            let b = img[i * 3 + 2];
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect())
}

/// Remove a rectangular region (for inpainting training).
/// Returns (corrupted image, mask) where mask[i] = 1.0 for removed pixels.
/// Region is clamped to image bounds.
pub fn mask_region(
    img: &[f32],
    w: usize,
    h: usize,
    channels: usize,
    x: usize,
    y: usize,
    rw: usize,
    rh: usize,
    fill: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let expected = w * h * channels;
    if img.len() != expected {
        return Err(format!("img len {} != {}", img.len(), expected));
    }
    if channels == 0 {
        return Err("channels must be > 0".into());
    }
    let mut out = img.to_vec();
    let mut mask = vec![0.0f32; w * h];
    let x1 = x.min(w);
    let y1 = y.min(h);
    let x2 = (x + rw).min(w);
    let y2 = (y + rh).min(h);
    for yy in y1..y2 {
        for xx in x1..x2 {
            let p = yy * w + xx;
            mask[p] = 1.0;
            for c in 0..channels {
                out[p * channels + c] = fill;
            }
        }
    }
    Ok((out, mask))
}

/// Additive uniform noise, severity in [0,1] scales the amplitude.
pub fn add_noise(
    img: &[f32],
    severity: f32,
    rng: &mut Rng,
) -> Result<Vec<f32>, String> {
    if !(0.0..=1.0).contains(&severity) {
        return Err(format!("severity {severity} outside [0,1]"));
    }
    let amp = severity * 0.5;
    Ok(img
        .iter()
        .map(|&v| v + (rng.next_f32() * 2.0 - 1.0) * amp)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_boxes() -> Vec<Box4> {
        vec![[10.0, 10.0, 20.0, 8.0], [10.0, 30.0, 20.0, 8.0]]
    }

    #[test]
    fn overlap_actually_overlaps() {
        let mut rng = Rng::new(1);
        let clean = two_boxes();
        let c = corrupt_layout(&clean, LayoutCorruption::Overlap, 1.0, 100.0, 100.0, &mut rng).unwrap();
        assert!(iou(&c.boxes[0], &c.boxes[1]) > 0.0, "boxes must overlap at severity 1");
        assert_eq!(c.labels.iter().sum::<f32>(), 2.0);
        // severity 0 should not overlap
        let c0 = corrupt_layout(&clean, LayoutCorruption::Overlap, 0.0, 100.0, 100.0, &mut rng).unwrap();
        assert_eq!(iou(&c0.boxes[0], &c0.boxes[1]), 0.0);
    }

    #[test]
    fn overflow_right_exceeds_page() {
        let mut rng = Rng::new(2);
        let clean = two_boxes();
        let c = corrupt_layout(&clean, LayoutCorruption::OverflowRight, 1.0, 100.0, 100.0, &mut rng).unwrap();
        let broken = c
            .boxes
            .iter()
            .zip(c.labels.iter())
            .find(|(_, l)| **l == 1.0)
            .unwrap()
            .0;
        assert!(broken[0] + broken[2] > 100.0, "right edge {} must exceed page", broken[0] + broken[2]);
    }

    #[test]
    fn overflow_bottom_exceeds_page() {
        let mut rng = Rng::new(3);
        let clean = two_boxes();
        let c = corrupt_layout(&clean, LayoutCorruption::OverflowBottom, 0.5, 100.0, 100.0, &mut rng).unwrap();
        let broken = c
            .boxes
            .iter()
            .zip(c.labels.iter())
            .find(|(_, l)| **l == 1.0)
            .unwrap()
            .0;
        assert!(broken[1] + broken[3] > 100.0);
    }

    #[test]
    fn dead_space_creates_gap() {
        let mut rng = Rng::new(4);
        // 4 boxes stacked with 10-unit gaps
        let clean: Vec<Box4> = (0..4).map(|i| [10.0, 10.0 + i as f32 * 20.0, 20.0, 8.0]).collect();
        let c = corrupt_layout(&clean, LayoutCorruption::DeadSpace, 1.0, 100.0, 100.0, &mut rng).unwrap();
        let max_y = c.boxes.iter().map(|b| b[1]).fold(f32::NEG_INFINITY, f32::max);
        let clean_max = clean.iter().map(|b| b[1]).fold(f32::NEG_INFINITY, f32::max);
        assert!(max_y > clean_max, "trailing boxes must be pushed down");
        assert!(c.labels.iter().sum::<f32>() >= 2.0);
    }

    #[test]
    fn squeeze_compresses_spread() {
        let mut rng = Rng::new(5);
        let clean: Vec<Box4> = (0..4).map(|i| [10.0, 10.0 + i as f32 * 20.0, 20.0, 8.0]).collect();
        let c = corrupt_layout(&clean, LayoutCorruption::Squeeze, 1.0, 100.0, 100.0, &mut rng).unwrap();
        let spread = c.boxes.iter().map(|b| b[1]).fold(f32::NEG_INFINITY, f32::max)
            - c.boxes.iter().map(|b| b[1]).fold(f32::INFINITY, f32::min);
        let clean_spread = 60.0;
        assert!(spread < clean_spread * 0.2, "spread {spread} should compress to <20% of {clean_spread}");
    }

    #[test]
    fn corrupt_layout_validates_input() {
        let mut rng = Rng::new(6);
        assert!(corrupt_layout(&[], LayoutCorruption::Overlap, 0.5, 100.0, 100.0, &mut rng).is_err());
        assert!(corrupt_layout(&two_boxes(), LayoutCorruption::Overlap, 1.5, 100.0, 100.0, &mut rng).is_err());
        assert!(corrupt_layout(&two_boxes(), LayoutCorruption::OverflowRight, 0.5, -1.0, 100.0, &mut rng).is_err());
        // single box can't overlap
        assert!(corrupt_layout(&[two_boxes()[0]], LayoutCorruption::Overlap, 0.5, 100.0, 100.0, &mut rng).is_err());
    }

    #[test]
    fn grayscale_luminance_formula() {
        // pure red -> 0.299
        let img = vec![1.0, 0.0, 0.0];
        let g = to_grayscale(&img, 1, 1).unwrap();
        assert!((g[0] - 0.299).abs() < 1e-6);
        // white -> 1.0
        let img = vec![1.0, 1.0, 1.0];
        let g = to_grayscale(&img, 1, 1).unwrap();
        assert!((g[0] - 1.0).abs() < 1e-6);
        assert!(to_grayscale(&[1.0], 1, 1).is_err());
    }

    #[test]
    fn mask_region_clamps_and_labels() {
        let w = 4;
        let img: Vec<f32> = (0..4 * 4 * 3).map(|i| i as f32).collect();
        let (out, mask) = mask_region(&img, w, 4, 3, 2, 2, 100, 100, -1.0).unwrap();
        // region clamped to (2..4, 2..4) = 4 pixels
        assert_eq!(mask.iter().sum::<f32>(), 4.0);
        assert_eq!(mask[2 * w + 2], 1.0);
        assert_eq!(mask[0], 0.0);
        // filled pixels
        assert_eq!(out[(2 * w + 2) * 3], -1.0);
        assert_eq!(out[0], 0.0); // untouched
    }

    #[test]
    fn noise_stays_bounded_by_severity() {
        let mut rng = Rng::new(7);
        let img = vec![0.5f32; 1000];
        let noisy = add_noise(&img, 1.0, &mut rng).unwrap();
        let max_dev = noisy
            .iter()
            .map(|v| (v - 0.5).abs())
            .fold(0.0f32, f32::max);
        assert!(max_dev <= 0.5 + 1e-6, "max deviation {max_dev}");
        assert!(max_dev > 0.1, "noise should actually perturb");
        assert!(add_noise(&img, 1.5, &mut rng).is_err());
    }
}
