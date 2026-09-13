//! Image <-> token-sequence conversion. An image is a grid of non-overlapping
//! patches; each patch becomes one token whose channels are the flattened
//! pixel values (ViT-style, but kept simple: no separate patch embedding
//! matrix — the model's input Linear does that).

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchGrid {
    pub img_w: usize,
    pub img_h: usize,
    pub patch: usize,
}

impl PatchGrid {
    pub fn new(img_w: usize, img_h: usize, patch: usize) -> Result<Self, String> {
        if patch == 0 {
            return Err("patch size must be > 0".into());
        }
        if img_w == 0 || img_h == 0 {
            return Err("image dimensions must be > 0".into());
        }
        if img_w % patch != 0 || img_h % patch != 0 {
            return Err(format!(
                "image {img_w}x{img_h} not divisible by patch size {patch}"
            ));
        }
        Ok(PatchGrid { img_w, img_h, patch })
    }

    pub fn grid_w(&self) -> usize {
        self.img_w / self.patch
    }

    pub fn grid_h(&self) -> usize {
        self.img_h / self.patch
    }

    pub fn n_tokens(&self) -> usize {
        self.grid_w() * self.grid_h()
    }
}

/// Convert image (row-major, `channels` interleaved: [r,g,b,r,g,b,...]) into
/// a (n_tokens, channels * patch * patch) token tensor, row-major patch order.
pub fn patchify_image(
    img: &[f32],
    grid: &PatchGrid,
    channels: usize,
) -> Result<Tensor, String> {
    let expected = grid.img_w * grid.img_h * channels;
    if img.len() != expected {
        return Err(format!("image len {} != expected {}", img.len(), expected));
    }
    let p = grid.patch;
    let gw = grid.grid_w();
    let gh = grid.grid_h();
    let token_dim = channels * p * p;
    let mut out = Tensor::zeros(gw * gh, token_dim);
    for gy in 0..gh {
        for gx in 0..gw {
            let tok = gy * gw + gx;
            let mut c = 0;
            for py in 0..p {
                for px in 0..p {
                    let x = gx * p + px;
                    let y = gy * p + py;
                    for ch in 0..channels {
                        out.data[tok * token_dim + c] = img[(y * grid.img_w + x) * channels + ch];
                        c += 1;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Inverse of [patchify_image].
pub fn unpatchify_image(
    tokens: &Tensor,
    grid: &PatchGrid,
    channels: usize,
) -> Result<Vec<f32>, String> {
    let token_dim = channels * grid.patch * grid.patch;
    if tokens.cols != token_dim {
        return Err(format!(
            "token width {} != expected {}",
            tokens.cols, token_dim
        ));
    }
    if tokens.rows != grid.n_tokens() {
        return Err(format!(
            "token count {} != expected {}",
            tokens.rows,
            grid.n_tokens()
        ));
    }
    let p = grid.patch;
    let gw = grid.grid_w();
    let mut img = vec![0.0f32; grid.img_w * grid.img_h * channels];
    for tok in 0..tokens.rows {
        let gx = tok % gw;
        let gy = tok / gw;
        let mut c = 0;
        for py in 0..p {
            for px in 0..p {
                let x = gx * p + px;
                let y = gy * p + py;
                for ch in 0..channels {
                    img[(y * grid.img_w + x) * channels + ch] = tokens.data[tok * token_dim + c];
                    c += 1;
                }
            }
        }
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_roundtrip() {
        // 4x4 image, 2 channels, patch 2 -> 4 tokens of 8 dims
        let img: Vec<f32> = (0..4 * 4 * 2).map(|i| i as f32).collect();
        let grid = PatchGrid::new(4, 4, 2).unwrap();
        let tokens = patchify_image(&img, &grid, 2).unwrap();
        assert_eq!(tokens.rows, 4);
        assert_eq!(tokens.cols, 8);
        let back = unpatchify_image(&tokens, &grid, 2).unwrap();
        assert_eq!(back, img);
    }

    #[test]
    fn patchify_rejects_odd_sizes() {
        assert!(PatchGrid::new(5, 5, 2).is_err());
        assert!(PatchGrid::new(4, 4, 0).is_err());
    }

    #[test]
    fn patch_contents_correct() {
        // 2x2 image, 1 channel, patch 2 -> single token [a, b, c, d]
        let img = vec![1.0, 2.0, 3.0, 4.0];
        let grid = PatchGrid::new(2, 2, 2).unwrap();
        let tokens = patchify_image(&img, &grid, 1).unwrap();
        assert_eq!(tokens.rows, 1);
        assert_eq!(tokens.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
