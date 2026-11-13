//! Multimodal image support: CLIP projector loading, image preprocessing, the Qwen3-VL encoder and prompt
//! position assignment.

pub mod image;
pub mod mmproj;
pub mod projector;
pub mod prompt;

pub use image::{PreprocessedImage, Rgb, decode_bytes, decode_path, load_image, smart_resize};
pub use mmproj::{ClipConfig, ClipModel, FfnKind, MergeKind, ProjectorKind, find_sibling, is_mmproj, validate_projection};
pub use projector::{VisionProjector, encode_image};
pub use prompt::{ImageSpan, Prompt, expand_image, expand_images, feed_prompt, next_text_position};
