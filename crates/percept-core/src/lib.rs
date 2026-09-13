//! # percept-core
//!
//! The "brain" of PerceptKit: a compact, dependency-light transformer stack for
//! building self-supervised perception systems — scorers, rankers, and dense
//! transformers — that runs natively and compiles to WebAssembly.
//!
//! Design goals:
//! - **Multimodal by unification.** Every task (document scoring, image
//!   colorization, inpainting, layout ranking) is expressed as a sequence of
//!   tokens with a small number of channels. Images are patchified; documents
//!   are already sequences. The same encoder serves all of them.
//! - **Autodiff-lite.** Each op has an explicit forward and backward. No graph
//!   building, no tape. The model owns its gradient buffers and the training
//!   loop is a plain function. This keeps the wasm binary small and the
//!   numerics auditable.
//! - **Testable.** Every backward pass is gradient-checked against finite
//!   differences; every training path has a convergence test on a synthetic
//!   task that must actually be learned, not just "run".

pub mod tensor;
pub mod rng;
pub mod layers;
pub mod model;
pub mod optim;
pub mod loss;
pub mod train;
pub mod patchify;
pub mod serialize;

pub use tensor::Tensor;
pub use model::{ModelConfig, PerceptModel};
pub use train::{Trainer, TrainerConfig};
pub use loss::{LossKind, PairwiseInput};
pub use patchify::{patchify_image, unpatchify_image, PatchGrid};
