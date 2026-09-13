//! Model serialization: a compact custom binary format (little-endian f32),
//! versioned, with a JSON config header. Designed for saving trained models
//! in the browser (download/upload, IndexedDB blob) and reusing them later.
//!
//! Layout:
//!   magic "PKM1" | u32 json_len | json config | f32 params...

use crate::model::{ModelConfig, PerceptModel};
use crate::rng::Rng;

pub const MAGIC: &[u8; 4] = b"PKM1";

pub fn to_bytes(model: &PerceptModel) -> Vec<u8> {
    let config_json = serde_json::to_string(&model.config).expect("config serialize");
    let mut out = Vec::with_capacity(4096);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(config_json.len() as u32).to_le_bytes());
    out.extend_from_slice(config_json.as_bytes());
    // Collect all params in deterministic order (same as param_refs()).
    for p in model.param_refs() {
        for v in p.data.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

pub fn from_bytes(bytes: &[u8]) -> Result<PerceptModel, String> {
    if bytes.len() < 8 {
        return Err("file too short".into());
    }
    if &bytes[0..4] != MAGIC {
        return Err("bad magic: not a PerceptKit model file".into());
    }
    let json_len = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if bytes.len() < 8 + json_len {
        return Err("truncated config header".into());
    }
    let config_json = std::str::from_utf8(&bytes[8..8 + json_len])
        .map_err(|_| "config not utf8".to_string())?;
    let config: ModelConfig =
        serde_json::from_str(config_json).map_err(|e| format!("bad config: {e}"))?;

    let mut rng = Rng::new(0); // deterministic; weights are overwritten below
    let mut model = PerceptModel::new(config, &mut rng)?;

    let mut data_start = 8 + json_len;
    let mut model_params = model.params();
    for (p, _) in model_params.iter_mut() {
        let n = p.len();
        if data_start + n * 4 > bytes.len() {
            return Err("truncated parameter data".into());
        }
        for i in 0..n {
            let off = data_start + i * 4;
            p.data[i] = f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]);
        }
        data_start += n * 4;
    }
    if data_start != bytes.len() {
        return Err(format!(
            "extra trailing bytes: expected end at {data_start}, got {}",
            bytes.len()
        ));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;
    use crate::Tensor;

    fn test_config() -> ModelConfig {
        ModelConfig {
            in_channels: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 2,
            d_ff: 32,
            out_channels: 3,
            max_seq_len: 64,
        }
    }

    #[test]
    fn roundtrip_preserves_weights_and_outputs() {
        let mut rng = Rng::new(5);
        let model = PerceptModel::new(test_config(), &mut rng).unwrap();
        let tokens = Tensor::from_rows((0..10).map(|_| {
            (0..4).map(|_| rng.next_f32()).collect::<Vec<f32>>()
        }).collect());
        let out1 = model.forward(&tokens).unwrap();

        let bytes = to_bytes(&model);
        let loaded = from_bytes(&bytes).unwrap();
        let out2 = loaded.forward(&tokens).unwrap();

        assert_eq!(out1.global_score.rows, out2.global_score.rows);
        for (a, b) in out1.global_score.data.iter().zip(out2.global_score.data.iter()) {
            assert_eq!(a, b);
        }
        assert_eq!(out1.dense.data, out2.dense.data);
    }

    #[test]
    fn rejects_garbage() {
        assert!(from_bytes(b"nope").is_err());
        assert!(from_bytes(b"PKM1").is_err());
        // valid header, truncated params
        let mut rng = Rng::new(1);
        let model = PerceptModel::new(test_config(), &mut rng).unwrap();
        let mut bytes = to_bytes(&model);
        bytes.truncate(bytes.len() - 10);
        assert!(from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut rng = Rng::new(2);
        let model = PerceptModel::new(test_config(), &mut rng).unwrap();
        let mut bytes = to_bytes(&model);
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(from_bytes(&bytes).is_err());
    }
}
