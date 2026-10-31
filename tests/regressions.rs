//! Regression tests for option handling and unusual input shapes.

#![cfg(feature = "cli")]

use benday_dots::image::{DynamicImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, RgbImage, Rgba};
use benday_dots::{Error, Options, PngCompression, convert, convert_dynamic, convert_file, convert_with_stages};

fn small(o: Options) -> Options {
    o.work_size(200).dots_across(20.0)
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("benday_reg_{}_{name}", std::process::id()))
}

fn noisy(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| Rgb([(x * 7 + y * 3) as u8, (x * 2 + y * 9) as u8, ((x * 5) ^ (y * 11)) as u8]))
}

#[test]
fn tiny_inputs_convert() {
    let two = RgbImage::from_fn(2, 1, |x, _| if x == 0 { Rgb([250, 20, 20]) } else { Rgb([10, 10, 240]) });
    let cases = [
        RgbImage::from_pixel(1, 1, Rgb([200, 100, 50])),
        two,
        noisy(1, 37),
        noisy(37, 1),
        noisy(2, 3),
        noisy(3, 3),
        noisy(1, 3000),
        noisy(3000, 1),
    ];
    for opts in [Options::comic(), Options::pop(), Options::comic().colors(Some(64))] {
        for img in &cases {
            let out = convert(img, &opts).unwrap_or_else(|e| panic!("{}x{}: {e}", img.width(), img.height()));
            assert_eq!(out.dimensions(), img.dimensions());
        }
    }
    let out = convert(&RgbImage::from_pixel(10, 10, Rgb([9, 9, 9])), &Options::comic().scale(0.001)).unwrap();
    assert_eq!(out.dimensions(), (1, 1));
}

#[test]
fn empty_input_is_an_error() {
    for (w, h) in [(0, 0), (0, 5), (5, 0)] {
        let err = convert(&RgbImage::new(w, h), &Options::comic()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "{w}x{h}: {err}");
        assert!(convert_with_stages(&RgbImage::new(w, h), &Options::pop()).is_err());
    }
}

#[test]
fn absurd_sizes_are_errors_not_aborts() {
    let img = noisy(10, 10);
    for scale in [1e7f32, f32::MAX, 1e5] {
        let err = convert(&img, &Options::comic().scale(scale)).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "scale {scale}: {err}");
    }
    assert!(convert(&img, &Options::comic().pitch(1e30)).is_err());
    assert!(convert(&img, &Options::comic().dots_across(1e-30)).is_err());
    assert!(convert(&img, &Options::comic().dots_across(f32::NAN)).is_err());
}

#[test]
fn hidden_knobs_are_validated() {
    let img = noisy(20, 20);
    for allowed in [vec![], vec![64], vec![200], vec![0, 63, 64]] {
        let mut o = small(Options::comic());
        o.allowed = Some(allowed.clone());
        let err = convert(&img, &o).unwrap_err();
        assert!(matches!(err, Error::InvalidOptions(_)), "{allowed:?}: {err}");
    }
    let mut o = Options::comic();
    o.hue_weight = f32::NAN;
    assert!(convert(&img, &o).is_err());
    let mut o = Options::comic();
    o.misregister = 1e30;
    assert!(convert(&img, &o).is_err());
}

#[test]
fn flat_dark_grey_is_not_solid_black() {
    for v in [60u8, 62, 64] {
        let img = RgbImage::from_pixel(80, 60, Rgb([v, v, v]));
        let out = convert_with_stages(&img, &small(Options::comic()).lines(false)).unwrap();
        let alpha = out.ink_alpha.expect("black mask exists");
        let black = alpha.as_raw().iter().filter(|&&a| a > 128).count();
        assert_eq!(black, 0, "flat RGB {v} came out with {black} black pixels");
    }
}

fn ramp() -> RgbImage {
    RgbImage::from_fn(256, 64, |x, _| Rgb([x as u8; 3]))
}

fn black_fraction(opts: &Options) -> f32 {
    let alpha = convert_with_stages(&ramp(), opts).unwrap().ink_alpha;
    alpha.map_or(0.0, |a| a.as_raw().iter().filter(|&&v| v > 128).count() as f32 / a.as_raw().len() as f32)
}

#[test]
fn explicit_blacks_can_lower_the_mask() {
    let base = small(Options::comic()).lines(false);
    let auto = black_fraction(&base);
    let fractions: Vec<f32> =
        [2.0, 5.0, 10.0, 20.0, 40.0].iter().map(|&b| black_fraction(&base.clone().blacks(b))).collect();
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "not monotone: {fractions:?}");
    assert!(fractions[1] < auto, "blacks(5) = {} should be below the automatic {auto}", fractions[1]);
    assert!(fractions[4] > fractions[0], "{fractions:?}");
    assert!(auto > 0.0);
}

#[test]
fn zero_blacks_disables_the_mask_without_exact_float_compare() {
    let base = small(Options::comic()).lines(false);
    for off in [0.0f32, -0.0] {
        let out = convert_with_stages(&ramp(), &base.clone().blacks(off)).unwrap();
        assert!(out.ink_alpha.is_none(), "blacks({off}) should leave no ink");
    }
    let with_lines = convert_with_stages(&ramp(), &small(Options::comic()).blacks(0.0)).unwrap();
    assert!(with_lines.ink_alpha.is_some());
}

#[test]
fn comic_with_pop_knobs_behaves_like_pop() {
    let img = noisy(120, 90);
    let pop = small(Options::pop());
    let mut c = small(Options::comic());
    c.dots_across = pop.dots_across;
    c.inks = pop.inks;
    c.misregister = pop.misregister;
    c.dot_gain = pop.dot_gain;
    c.grain = pop.grain;
    c.l_weight = pop.l_weight;
    c.hue_weight = pop.hue_weight;
    c.bilateral_passes = pop.bilateral_passes;
    c.mode_radius_mul = pop.mode_radius_mul;
    c.min_area_mul = pop.min_area_mul;
    c.line_dilate_mul = pop.line_dilate_mul;
    c.line_area_mul = pop.line_area_mul;
    c.angles = pop.angles;
    c.allowed = pop.allowed.clone();
    c.keep_dark_codes = pop.keep_dark_codes;
    c.solid_black_l = pop.solid_black_l;
    assert_eq!(convert(&img, &c).unwrap(), convert(&img, &pop).unwrap());
}

#[test]
fn convert_dynamic_accepts_every_colour_type() {
    let opts = small(Options::pop());
    let (w, h) = (40u32, 30u32);
    let images = [
        DynamicImage::ImageLuma8(GrayImage::from_fn(w, h, |x, _| Luma([(x * 6) as u8]))),
        DynamicImage::ImageLumaA16(ImageBuffer::from_fn(w, h, |x, y| LumaA([(x * 1500) as u16, (y * 2000) as u16]))),
        DynamicImage::ImageRgba8(ImageBuffer::from_fn(w, h, |x, y| Rgba([(x * 6) as u8, (y * 8) as u8, 90, 128]))),
        DynamicImage::ImageRgb16(ImageBuffer::from_fn(w, h, |x, y| Rgb([(x * 1500) as u16, (y * 2000) as u16, 9000]))),
        DynamicImage::ImageRgba32F(ImageBuffer::from_fn(w, h, |x, _| Rgba([x as f32 / 40.0, 0.5, 0.2, 1.0]))),
    ];
    for img in &images {
        let out = convert_dynamic(img, &opts).unwrap_or_else(|e| panic!("{:?}: {e}", img.color()));
        assert_eq!(out.dimensions(), (w, h));
    }
}

/// A JPEG carrying an EXIF Orientation tag, built by splicing an APP1 segment behind the SOI marker.
fn jpeg_with_orientation(img: &RgbImage, orientation: u16) -> Vec<u8> {
    let mut plain = Vec::new();
    benday_dots::image::codecs::jpeg::JpegEncoder::new_with_quality(&mut plain, 95)
        .encode_image(&DynamicImage::ImageRgb8(img.clone()))
        .unwrap();
    let mut tiff = b"II*\0\x08\0\0\0".to_vec();
    tiff.extend_from_slice(&1u16.to_le_bytes());
    tiff.extend_from_slice(&0x0112u16.to_le_bytes());
    tiff.extend_from_slice(&3u16.to_le_bytes());
    tiff.extend_from_slice(&1u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(&tiff);
    let mut out = plain[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&plain[2..]);
    out
}

#[test]
fn exif_orientation_is_applied_by_convert_file() {
    let img = RgbImage::from_fn(48, 24, |x, _| if x < 24 { Rgb([230, 30, 30]) } else { Rgb([30, 30, 230]) });
    let (input, output) = (scratch("rot.jpg"), scratch("rot.png"));
    std::fs::write(&input, jpeg_with_orientation(&img, 6)).unwrap();
    let decoded = benday_dots::io::decode(&input).unwrap();
    assert_eq!(decoded.dimensions(), (24, 48), "orientation 6 turns the image upright");
    assert!(decoded.get_pixel(12, 5)[0] > 150, "left half was red, so it ends up on top");
    assert!(decoded.get_pixel(12, 40)[2] > 150);
    convert_file(&input, &output, &small(Options::pop())).unwrap();
    assert_eq!(benday_dots::image::image_dimensions(&output).unwrap(), (24, 48));
    std::fs::write(&input, jpeg_with_orientation(&img, 1)).unwrap();
    assert_eq!(benday_dots::io::decode(&input).unwrap().dimensions(), (48, 24));
    std::fs::remove_file(&input).unwrap();
    std::fs::remove_file(&output).unwrap();
}

#[test]
fn small_outputs_report_the_pitch_clamp() {
    let img = noisy(60, 40);
    let out = convert_with_stages(&img, &Options::comic()).unwrap();
    assert!(out.pitch_clamped && out.pitch == benday_dots::MIN_PITCH);
    let out = convert_with_stages(&img, &Options::comic().pitch(9.0)).unwrap();
    assert!(!out.pitch_clamped && out.pitch == 9.0);
}

#[test]
fn error_message_appears_once_in_the_chain() {
    let err = convert_file(scratch("missing.png"), scratch("out.png"), &Options::comic()).unwrap_err();
    let mut chain = vec![err.to_string()];
    let mut src = std::error::Error::source(&err);
    while let Some(e) = src {
        chain.push(e.to_string());
        src = e.source();
    }
    assert_eq!(chain.len(), 1, "{chain:?}");
    assert!(chain[0].contains("missing.png"), "{}", chain[0]);
}

#[test]
fn every_png_mode_works_through_convert_file() {
    let (input, output) = (scratch("in.png"), scratch("out2.png"));
    noisy(90, 60).save(&input).unwrap();
    for mode in [PngCompression::Fast, PngCompression::Fastest, PngCompression::None] {
        let opts = small(Options::comic()).png_compression(mode);
        convert_file(&input, &output, &opts).unwrap();
        let direct = convert(&noisy(90, 60), &opts).unwrap();
        assert_eq!(benday_dots::image::open(&output).unwrap().to_rgb8(), direct, "{mode:?}");
    }
    std::fs::remove_file(&input).unwrap();
    std::fs::remove_file(&output).unwrap();
}

#[test]
fn bad_output_extension_fails_before_the_input_is_read() {
    let err = convert_file(scratch("never-read.png"), scratch("out.xyz"), &Options::comic()).unwrap_err();
    assert!(err.to_string().contains("out.xyz"), "{err}");
}

#[cfg(feature = "cli")]
mod cli {
    use std::process::Command;

    fn run(args: &[&str]) -> (bool, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_benday-dots")).args(args).output().unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    }

    #[test]
    fn colors_out_of_range_is_a_usage_error() {
        for bad in ["0", "99"] {
            let (ok, err) = run(&["a.png", "b.png", "--colors", bad]);
            assert!(!ok && err.contains("--colors"), "{bad}: {err}");
        }
        let (ok, err) = run(&["a.png", "b.png", "--palette", "benday", "--colors", "4"]);
        assert!(!ok && err.contains("--colors"), "{err}");
    }

    #[test]
    fn bad_extension_and_missing_input_name_the_file() {
        let (ok, err) = run(&["does-not-exist.png", "out.xyz"]);
        assert!(!ok && err.contains("out.xyz") && !err.contains("does-not-exist"), "{err}");
        let (ok, err) = run(&["does-not-exist.png", "out.png"]);
        assert!(!ok && err.contains("does-not-exist.png"), "{err}");
    }

    #[test]
    fn pitch_clamp_warning_is_always_printed() {
        let dir = std::env::temp_dir().join(format!("benday_reg_cli_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (input, output) = (dir.join("in.png"), dir.join("out.png"));
        super::noisy(60, 40).save(&input).unwrap();
        let (ok, err) = run(&[input.to_str().unwrap(), output.to_str().unwrap(), "--pitch", "2"]);
        assert!(ok && err.contains("pitch"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Options>();
    assert_send_sync::<benday_dots::Output>();
    assert_send_sync::<Error>();
}

fn pixel_types(w: u32, h: u32) -> Vec<DynamicImage> {
    let v = |x: u32, y: u32| (x * 7 + y * 13) % 251;
    vec![
        DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| Rgb([v(x, y) as u8, (v(y, x) / 2) as u8, 90]))),
        DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, y| Luma([v(x, y) as u8]))),
        DynamicImage::ImageLumaA8(ImageBuffer::from_fn(w, h, |x, y| LumaA([v(x, y) as u8, 200]))),
        DynamicImage::ImageRgba8(ImageBuffer::from_fn(w, h, |x, y| Rgba([v(x, y) as u8, 40, v(y, x) as u8, 128]))),
        DynamicImage::ImageRgb16(ImageBuffer::from_fn(w, h, |x, y| Rgb([(v(x, y) * 250) as u16, 9000, 3000]))),
        DynamicImage::ImageRgb32F(ImageBuffer::from_fn(w, h, |x, y| {
            let n = v(x, y) as f32 / 250.0;
            match (x + y) % 11 {
                0 => Rgb([f32::NAN, n, 0.5]),
                1 => Rgb([2.0, n, -1.0]),
                _ => Rgb([n, 1.0 - n, 0.3]),
            }
        })),
    ]
}

#[test]
fn input_shape_sweep_never_panics_and_keeps_size() {
    let shapes = [(1, 1), (2, 1), (1, 300), (7, 1), (65, 257), (257, 65), (513, 3), (3, 513)];
    // Each pixel type meets a different option set per shape; the palette lookup build dominates the run time.
    let configs: [(&str, Options, u32); 4] = [
        ("comic", Options::comic().work_size(64), 1),
        ("pop", Options::pop().work_size(64), 1),
        ("scale 3", Options::comic().scale(3.0).work_size(64), 3),
        ("work 1", Options::pop().work_size(1), 1),
    ];
    for (n, (w, h)) in shapes.into_iter().enumerate() {
        for (i, img) in pixel_types(w, h).into_iter().enumerate() {
            let (name, opts, k) = &configs[(n + i) % configs.len()];
            let out = convert_dynamic(&img, opts).unwrap_or_else(|e| panic!("{w}x{h} {:?} {name}: {e}", img.color()));
            assert_eq!(out.dimensions(), (w * k, h * k), "{w}x{h} {:?} {name}", img.color());
        }
    }
}
