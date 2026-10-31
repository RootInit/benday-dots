//! Turn photographs into Silver-Age comic prints: flat regions in the 64 historical YRB tint codes, screened
//! as Ben-Day dots on aged paper, with solid black ink lines and spotted blacks on top.
//!
//! # Example
//!
//! ```
//! use benday_dots::{Options, convert, image::{Rgb, RgbImage}};
//!
//! let photo = RgbImage::from_fn(64, 48, |x, y| Rgb([(x * 4) as u8, (y * 5) as u8, 128]));
//! let opts = Options::pop().dots_across(8.0).work_size(64);
//! let print = convert(&photo, &opts)?;
//! assert_eq!(print.dimensions(), photo.dimensions());
//! # Ok::<(), benday_dots::Error>(())
//! ```
//!
//! Converting files (EXIF orientation is applied):
//!
//! ```no_run
//! # #[cfg(feature = "io")] {
//! use benday_dots::{Options, Style, convert_file};
//!
//! let opts = Options::from_style(Style::Comic).dots_across(300.0).seed(7);
//! convert_file("photo.jpg", "print.png", &opts)?;
//! # }
//! # Ok::<(), benday_dots::Error>(())
//! ```
//!
//! # Features
//!
//! `io` (default) adds PNG/JPEG file support: [`convert_file`] and the `io` module. `cli` (default) builds the
//! `benday-dots` binary with `clap` and implies `io`. Use `default-features = false` to work on in-memory pixels only.
//!
//! The lower-level pipeline stages live in hidden modules; [`convert_with_stages`] exposes their products.

#![warn(missing_docs)]

#[doc(hidden)]
pub mod cc;
#[doc(hidden)]
pub mod chart;
#[cfg(feature = "cli")]
#[doc(hidden)]
pub mod cli;
#[doc(hidden)]
pub mod codes;
#[doc(hidden)]
pub mod composite;
#[cfg(feature = "io")]
mod deflate;
mod error;
#[doc(hidden)]
pub mod ink;
#[cfg(feature = "io")]
#[doc(hidden)]
pub mod io;
#[doc(hidden)]
pub mod lines;
#[doc(hidden)]
pub mod morph;
#[doc(hidden)]
pub mod noise;
mod options;
#[doc(hidden)]
pub mod pipeline;
#[cfg(feature = "io")]
mod pngwrite;
#[doc(hidden)]
pub mod quantize;
#[doc(hidden)]
pub mod render;
#[doc(hidden)]
pub mod screen;
#[doc(hidden)]
pub mod smooth;
#[doc(hidden)]
pub mod types;

#[cfg(test)]
mod testutil;

pub use error::Error;
pub use image;
#[cfg(feature = "io")]
pub use io::PngCompression;
pub use options::{InkChoice, MAX_PITCH, MIN_PITCH, Options, PaperChoice, Style};
pub use pipeline::{MAX_OUTPUT_PIXELS, Output};
pub use types::{InkSet, Plane};

use image::{DynamicImage, RgbImage};
#[cfg(feature = "io")]
use std::path::Path;

/// Convert an image to a Ben-Day print the same size times `opts.scale`.
///
/// # Errors
/// [`Error::InvalidOptions`] for an out-of-range option, [`Error::InvalidInput`] for an image with no pixels or an
/// output larger than [`MAX_OUTPUT_PIXELS`].
pub fn convert(img: &RgbImage, opts: &Options) -> Result<RgbImage, Error> {
    pipeline::run_print(img, opts)
}

/// Like [`convert`] for any decoded image (grey, 16-bit, ...). Alpha is discarded: transparent pixels keep their
/// colour channels, which are often black.
///
/// # Errors
/// As [`convert`].
pub fn convert_dynamic(img: &DynamicImage, opts: &Options) -> Result<RgbImage, Error> {
    convert(&img.to_rgb8(), opts)
}

/// Like [`convert`] for tightly packed RGB8 bytes (`width * height * 3`, row-major).
///
/// Returns `(width, height, pixels)` of the output, also tightly packed RGB8.
///
/// # Errors
/// [`Error::InvalidInput`] when `rgb.len()` does not match the dimensions, otherwise as [`convert`].
pub fn convert_raw(width: u32, height: u32, rgb: &[u8], opts: &Options) -> Result<(u32, u32, Vec<u8>), Error> {
    let expected = u128::from(width) * u128::from(height) * 3;
    if rgb.len() as u128 != expected {
        return Err(Error::InvalidInput(format!("{width}x{height} RGB8 needs {expected} bytes, got {}", rgb.len())));
    }
    let img = RgbImage::from_raw(width, height, rgb.to_vec())
        .ok_or_else(|| Error::InvalidInput("pixel buffer does not fit the dimensions".into()))?;
    let out = convert(&img, opts)?;
    let (w, h) = out.dimensions();
    Ok((w, h, out.into_raw()))
}

/// Like [`convert`], but returns the code map, ink alpha and pitch along with the image.
///
/// # Errors
/// As [`convert`].
pub fn convert_with_stages(img: &RgbImage, opts: &Options) -> Result<Output, Error> {
    pipeline::run_image(img, opts)
}

/// Read `input` (EXIF orientation applied), convert it and write `output`; the format follows its extension,
/// which is checked before the input is read.
///
/// # Errors
/// As [`convert`], plus [`Error::Image`] or [`Error::Io`] (naming the file) when reading or writing fails or the
/// output extension is not a writable image format.
#[cfg(feature = "io")]
pub fn convert_file(input: impl AsRef<Path>, output: impl AsRef<Path>, opts: &Options) -> Result<(), Error> {
    let report = pipeline::run(input.as_ref(), output.as_ref(), opts)?;
    pipeline::warn_clamp(opts, report.pitch_clamped);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> (RgbImage, Options) {
        let img = RgbImage::from_fn(40, 30, |x, y| image::Rgb([(x * 6) as u8, (y * 8) as u8, 100]));
        (img, Options::pop().dots_across(6.0).work_size(40))
    }

    #[test]
    fn raw_matches_convert() {
        let (img, opts) = small();
        let want = convert(&img, &opts).unwrap();
        let (w, h, raw) = convert_raw(40, 30, img.as_raw(), &opts).unwrap();
        assert_eq!((w, h), want.dimensions());
        assert_eq!(raw, want.into_raw());
    }

    #[test]
    fn raw_rejects_wrong_length() {
        let (_, opts) = small();
        for len in [0, 40 * 30 * 3 - 1, 40 * 30 * 3 + 1] {
            let r = convert_raw(40, 30, &vec![0; len], &opts);
            assert!(matches!(r, Err(Error::InvalidInput(_))), "len {len}");
        }
        assert!(matches!(convert_raw(u32::MAX, u32::MAX, &[], &opts), Err(Error::InvalidInput(_))));
    }
}
