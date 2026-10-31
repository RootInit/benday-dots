//! Image file decode and save with the file name in every error.

use crate::error::Error;
use image::error::{DecodingError, EncodingError, ImageFormatHint};
use image::{DynamicImage, ImageDecoder, ImageError, ImageFormat, ImageReader, RgbImage};
use std::io::{BufWriter, Write};
use std::path::Path;

/// PNG encoder effort for [`save`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum PngCompression {
    /// Fast deflate with adaptive filtering: the smallest files of the three.
    #[default]
    Fast,
    /// Fast deflate with the `Up` filter: less filtering work, files a few percent larger.
    Fastest,
    /// Stored deflate blocks: quickest, and about three times the size.
    None,
}

fn named_io(e: &std::io::Error, name: &std::path::Display<'_>) -> Error {
    Error::Io(std::io::Error::new(e.kind(), format!("{name}: {e}")))
}

fn at_path(path: &Path, e: Error, decoding: bool) -> Error {
    let name = path.display();
    match e {
        Error::Io(e) | Error::Image(ImageError::IoError(e)) => named_io(&e, &name),
        Error::Image(e) if decoding => {
            Error::Image(ImageError::Decoding(DecodingError::new(ImageFormatHint::Unknown, format!("{name}: {e}"))))
        }
        Error::Image(e) => {
            Error::Image(ImageError::Encoding(EncodingError::new(ImageFormatHint::Unknown, format!("{name}: {e}"))))
        }
        other => other,
    }
}

/// Decode an image file to RGB8 with its EXIF orientation applied.
///
/// # Errors
/// Fails if the file cannot be read or decoded; the message names the file.
pub fn decode(path: &Path) -> Result<RgbImage, Error> {
    let run = || -> Result<RgbImage, Error> {
        let mut dec = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
        let orientation = dec.orientation()?;
        let mut img = DynamicImage::from_decoder(dec)?;
        img.apply_orientation(orientation);
        Ok(img.into_rgb8())
    };
    run().map_err(|e| at_path(path, e, true))
}

/// Check that [`save`] can write a file with this name, so a bad extension fails before any work is done.
///
/// # Errors
/// Fails if the extension is missing, unknown, or names a format this build cannot encode.
pub fn check_output(path: &Path) -> Result<(), Error> {
    let format = ImageFormat::from_path(path).map_err(|e| at_path(path, Error::Image(e), false))?;
    if !format.writing_enabled() {
        return Err(Error::InvalidInput(format!("{}: cannot write {format:?} files", path.display())));
    }
    Ok(())
}

/// Save `img` to `path`; the format follows the extension. `png` only applies to `.png` files; other formats use
/// the `image` crate's defaults.
///
/// # Errors
/// Fails if the extension is unknown or the file cannot be written; the message names the file.
pub fn save(img: &RgbImage, path: &Path, png: PngCompression) -> Result<(), Error> {
    let is_png = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png"));
    let run = || -> Result<(), Error> {
        if !is_png {
            return Ok(img.save(path)?);
        }
        let mut out = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        crate::pngwrite::write_rgb8(&mut out, img, png)?;
        out.flush()?;
        Ok(())
    };
    run().map_err(|e| at_path(path, e, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("benday_io_{}_{name}", std::process::id()))
    }

    const MODES: [PngCompression; 3] = [PngCompression::Fast, PngCompression::Fastest, PngCompression::None];

    fn round_trip(img: &RgbImage, name: &str) {
        for (i, mode) in MODES.into_iter().enumerate() {
            let p = tmp(&format!("{name}{i}.png"));
            save(img, &p, mode).unwrap();
            let back = decode(&p);
            let bytes = std::fs::read(&p).unwrap();
            std::fs::remove_file(&p).unwrap();
            assert!(back.unwrap() == *img, "{name} {mode:?}");
            let idats = bytes.windows(4).filter(|w| w == b"IDAT").count();
            assert!(idats >= 1, "{name} {mode:?}");
        }
    }

    #[test]
    fn png_round_trips_every_mode() {
        let img = RgbImage::from_fn(97, 61, |x, y| image::Rgb([(x * 5) as u8, (y * 3) as u8, ((x ^ y) * 7) as u8]));
        round_trip(&img, "rt");
    }

    #[test]
    fn png_round_trips_sizes_around_the_band_height() {
        for (w, h) in [(1u32, 1u32), (1, 65), (7, 130), (97, 61), (5, 64), (5, 128), (3, 129)] {
            let noisy = RgbImage::from_fn(w, h, |x, y| {
                let n = crate::noise::hash_u64(u64::from(x), u64::from(y), 9);
                image::Rgb([n as u8, (n >> 8) as u8, (x * 40 + y) as u8])
            });
            round_trip(&noisy, &format!("n{w}x{h}"));
            round_trip(&RgbImage::new(w, h), &format!("z{w}x{h}"));
            round_trip(&RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255])), &format!("w{w}x{h}"));
            let ramp = RgbImage::from_fn(w, h, |x, y| image::Rgb([(x + y) as u8, y as u8, (x * 2) as u8]));
            round_trip(&ramp, &format!("r{w}x{h}"));
        }
    }

    #[test]
    fn large_png_has_one_idat_per_band_and_round_trips() {
        let img = RgbImage::from_fn(4032, 3024, |x, y| {
            let n = crate::noise::hash_u64(u64::from(x), u64::from(y), 4);
            image::Rgb([n as u8, (n >> 8) as u8, (n >> 16) as u8])
        });
        let p = tmp("big.png");
        save(&img, &p, PngCompression::Fast).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert!(bytes.windows(4).filter(|w| w == b"IDAT").count() >= 3024 / 64);
        assert!(decode(&p).unwrap() == img);
        std::fs::remove_file(&p).unwrap();
        for mode in [PngCompression::Fastest, PngCompression::None] {
            save(&img, &p, mode).unwrap();
            assert!(decode(&p).unwrap() == img, "{mode:?}");
            std::fs::remove_file(&p).unwrap();
        }
    }

    #[test]
    fn other_extensions_use_image_defaults() {
        let img = RgbImage::from_fn(16, 16, |x, y| image::Rgb([x as u8 * 8, y as u8 * 8, 0]));
        let p = tmp("o.jpg");
        save(&img, &p, PngCompression::None).unwrap();
        assert_eq!(decode(&p).unwrap().dimensions(), (16, 16));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn errors_name_the_file() {
        let missing = tmp("missing.png");
        let msg = decode(&missing).unwrap_err().to_string();
        assert!(msg.contains("benday_io_") && msg.contains("missing.png"), "{msg}");
        let img = RgbImage::new(2, 2);
        let bad_dir = tmp("nodir").join("x.png");
        let msg = save(&img, &bad_dir, PngCompression::Fast).unwrap_err().to_string();
        assert!(msg.contains("nodir"), "{msg}");
        let bad_ext = tmp("x.xyz");
        let msg = save(&img, &bad_ext, PngCompression::Fast).unwrap_err().to_string();
        assert!(msg.contains("x.xyz"), "{msg}");
        let garbage = tmp("garbage.png");
        std::fs::write(&garbage, b"not an image").unwrap();
        let msg = decode(&garbage).unwrap_err().to_string();
        std::fs::remove_file(&garbage).unwrap();
        assert!(msg.contains("garbage.png"), "{msg}");
    }

    #[test]
    fn output_extension_is_checked_up_front() {
        assert!(check_output(Path::new("a.png")).is_ok());
        assert!(check_output(Path::new("a.JPG")).is_ok());
        assert!(check_output(Path::new("a.xyz")).is_err());
        assert!(check_output(Path::new("noext")).is_err());
    }

    fn write_synthetic_jpeg(path: &Path, w: u32, h: u32) {
        let img = RgbImage::from_fn(w, h, |x, y| {
            let n = crate::noise::hash_u64(u64::from(x), u64::from(y), 3);
            image::Rgb([(x * 255 / w) as u8, (y * 255 / h) as u8, 90 + (n & 15) as u8])
        });
        let mut f = std::fs::File::create(path).unwrap();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 90).encode_image(&img).unwrap();
    }

    #[test]
    fn jpeg_decodes() {
        let p = tmp("synthetic.jpg");
        write_synthetic_jpeg(&p, 64, 48);
        let img = decode(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(img.dimensions(), (64, 48));
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_io() {
        use crate::testutil::best_of;
        let p = tmp("bench.jpg");
        write_synthetic_jpeg(&p, 4000, 3000);
        let (w, c) = best_of(3, || decode(&p).unwrap());
        eprintln!("decode: wall {w:.1} ms, cpu {c:.1} ms");
        let img = decode(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        let noisy = RgbImage::from_fn(img.width(), img.height(), |x, y| {
            let h = crate::noise::hash_u64(x as u64, y as u64, 1);
            let p = img.get_pixel(x, y);
            image::Rgb([p[0].saturating_add((h & 7) as u8), p[1].saturating_add((h >> 3 & 7) as u8), p[2]])
        });
        for (name, im) in [("photo", &img), ("noisy", &noisy)] {
            for mode in [PngCompression::Fast, PngCompression::Fastest, PngCompression::None] {
                let out = tmp("bench.png");
                let (w, c) = best_of(3, || save(im, &out, mode).unwrap());
                let size = std::fs::metadata(&out).unwrap().len();
                std::fs::remove_file(&out).unwrap();
                eprintln!("save {name} {mode:?}: wall {w:.1} ms, cpu {c:.1} ms, {:.1} MB", size as f64 / 1e6);
            }
        }
    }
}
