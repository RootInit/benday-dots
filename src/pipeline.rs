use crate::codes::{LEVELS, PaletteLut, RenderedPalette, all_codes};
use crate::composite::separation_preview;
use crate::error::Error;
use crate::ink::InkModel;
#[cfg(feature = "io")]
use crate::io;
use crate::lines::{
    BlackParams, MaskParams, MaskScale, XdogParams, black_field, black_mask_work, line_mask_work, xdog_field,
};
use crate::options::{MAX_PITCH, Options};
use crate::quantize::{
    Boost, boost_from_stats, boost_l, boost_plane, kmeans_codes, lifted, lightness_stats, merge_small_regions,
    mode_filter, repick_regions, snap_work, unboost_l,
};
use crate::render::{RenderInput, Rendered, render};
use crate::screen::{ScreenConfig, ScreenParams, screen};
use crate::smooth::{BilateralParams, lab_to_rgb, smooth_rgb, to_working};
use crate::types::{IndexMap, LabPlane, TintField};
use image::{GrayImage, RgbImage};
use std::path::Path;
use std::time::Instant;

/// Codes rendering darker than this (full overprints: YRB, Y3RB, Y2RB) are left out of the palette: they print as
/// murky olive, so those tones are decided by the black mask instead, and the fill under it is RB navy.
const DARK_CODE_L: f32 = 12.0;
/// Mode-filter radius in output px at the reference resolution.
const MODE_RADIUS: f32 = 2.0;
/// Smallest region kept by the merge, as a fraction of the output area.
const MIN_REGION_AREA_FRAC: f64 = 0.00002;
/// Rounds of small-region merging.
const MERGE_PASSES: u32 = 3;
/// The automatic black level is at most this stretched L*.
const AUTO_BLACK_CAP: f32 = 15.0;
/// Above this work/output ratio the ink mask gets a 3x3 box blur, because bilinear upsampling only
/// anti-aliases it when the work grid is clearly coarser than the output.
const INK_BOX_MIN_RATIO: f32 = 0.75;
/// Largest output, in pixels, a conversion will allocate for.
pub const MAX_OUTPUT_PIXELS: u64 = 1 << 28;

type Res<T> = Result<T, Error>;

/// A finished conversion together with its intermediate products.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Output {
    /// The rendered print, `scale` times the input size.
    pub image: RgbImage,
    /// Per-pixel YRB code chosen for every output pixel: `c*16 + m*4 + y`, each level 0 (none) to 3 (solid).
    pub index: IndexMap,
    /// Black-ink coverage (lines and spotted blacks), 255 = solid ink; `None` when neither is enabled.
    pub ink_alpha: Option<GrayImage>,
    /// Dot pitch actually used, in output pixels.
    pub pitch: f32,
    /// True when the requested pitch was below [`MIN_PITCH`](crate::MIN_PITCH) and was raised to it.
    pub pitch_clamped: bool,
}

#[cfg(feature = "io")]
/// What [`run`] tells its caller about the conversion it did.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    /// Dot pitch actually used, in output pixels.
    pub pitch: f32,
    /// True when the requested pitch was below the minimum and was raised.
    pub pitch_clamped: bool,
}

struct Inner {
    image: RgbImage,
    index: Option<IndexMap>,
    alpha: Option<GrayImage>,
    pitch: f32,
    pitch_clamped: bool,
}

/// Prints one `name  ms` line per stage when verbose.
pub(crate) struct Timer {
    verbose: bool,
    t: Instant,
}

impl Timer {
    pub(crate) fn new(verbose: bool) -> Self {
        Timer { verbose, t: Instant::now() }
    }

    pub(crate) fn lap(&mut self, stage: &str) {
        if self.verbose {
            eprintln!("{stage:<12} {:>7.1} ms", self.t.elapsed().as_secs_f64() * 1e3);
        }
        self.t = Instant::now();
    }
}

#[cfg(feature = "io")]
fn dump<P>(dir: Option<&Path>, name: &str, img: &image::ImageBuffer<P, Vec<P::Subpixel>>) -> Res<()>
where
    P: image::Pixel + image::PixelWithColorType,
    [P::Subpixel]: image::EncodableLayout,
{
    if let Some(dir) = dir {
        img.save(dir.join(name))?;
    }
    Ok(())
}

#[cfg(not(feature = "io"))]
fn dump<P: image::Pixel>(_: Option<&Path>, _: &str, _: &image::ImageBuffer<P, Vec<P::Subpixel>>) -> Res<()> {
    Ok(())
}

#[cfg(feature = "io")]
fn dump_dir(opts: &Options) -> Option<&Path> {
    opts.dump_dir.as_deref()
}

#[cfg(not(feature = "io"))]
fn dump_dir(_: &Options) -> Option<&Path> {
    None
}

fn output_dims(w: u32, h: u32, scale: f32) -> Res<(u32, u32)> {
    let dim = |n: u32| (f64::from(n) * f64::from(scale)).round().max(1.0);
    let (ow, oh) = (dim(w), dim(h));
    if !(ow.is_finite() && oh.is_finite() && ow * oh <= MAX_OUTPUT_PIXELS as f64) {
        return Err(Error::InvalidInput(format!(
            "scale {scale} on a {w}x{h} image gives {ow}x{oh} px; the limit is {} megapixels",
            MAX_OUTPUT_PIXELS >> 20
        )));
    }
    Ok((ow as u32, oh as u32))
}

/// Union of the line and black masks, box-blurred when the work grid is nearly as fine as the output.
fn ink_mask(lines: Option<Vec<u8>>, blacks: Option<Vec<u8>>, w: u32, h: u32, f: f32) -> Option<Vec<u8>> {
    let mut ink = match (lines, blacks) {
        (None, None) => return None,
        (Some(m), None) | (None, Some(m)) => m,
        (Some(mut a), Some(b)) => {
            a.iter_mut().zip(&b).for_each(|(a, &b)| *a = (*a).max(b));
            a
        }
    };
    if f > INK_BOX_MIN_RATIO {
        ink = crate::morph::box3(&ink, w, h);
    }
    Some(ink)
}

/// Black-mask and line thresholds, in stretched L* (`black`, `None` = no black mask) and input L* / 100 (`line`).
struct Thresholds {
    line: f32,
    black: Option<f32>,
}

fn thresholds(opts: &Options, boost: &Boost, p08: f32) -> Thresholds {
    match opts.blacks {
        Some(u) if u > 0.0 => {
            let stretched = boost_l(u, boost);
            Thresholds { line: unboost_l(stretched, boost).max(0.0) / 100.0, black: Some(stretched / 100.0) }
        }
        Some(_) => Thresholds { line: 0.0, black: None },
        None => {
            let auto = boost_l(p08, boost).min(lifted(AUTO_BLACK_CAP, boost));
            Thresholds {
                line: unboost_l(auto, boost).max(0.0) / 100.0,
                black: Some(auto.max(lifted(opts.solid_black_l, boost)) / 100.0),
            }
        }
    }
}

/// Region codes on the work grid: snap to the palette, mode filter, merge small regions, repick large ones.
fn regions(bw: &LabPlane, lut: &PaletteLut, opts: &Options, g: &MaskScale, out_w: u32, out_h: u32) -> IndexMap {
    let mode_r = (MODE_RADIUS * g.s * opts.mode_radius_mul * g.f).round() as u32;
    let out_area = f64::from(out_w) * f64::from(out_h);
    let min_area =
        ((MIN_REGION_AREA_FRAC * f64::from(opts.min_area_mul) * out_area * f64::from(g.f * g.f)).round() as u32).max(1);
    let mut idx = mode_filter(&snap_work(bw, lut), mode_r, 1);
    merge_small_regions(&mut idx, min_area, MERGE_PASSES);
    if opts.repick_mul > 0.0 {
        let big = (min_area as f32 * opts.repick_mul).round() as u32;
        repick_regions(&mut idx, bw, lut, opts.l_weight, big);
    }
    idx
}

/// Line and black masks on the work grid, united into one ink mask. Dumps the two masks when `dump_dir` is set.
fn ink_masks(
    sm: &LabPlane,
    boost: &Boost,
    p08: f32,
    opts: &Options,
    g: &MaskScale,
    (work_w, work_h): (u32, u32),
    tm: &mut Timer,
) -> Res<Option<Vec<u8>>> {
    let th = thresholds(opts, boost, p08);
    let line_mask = opts.lines.then(|| {
        let xdog = xdog_field(sm, &XdogParams::default());
        line_mask_work(&xdog, th.line, &MaskParams::scaled(g, opts.line_area_mul, opts.line_dilate_mul))
    });
    tm.lap("lines");
    let black_mask = th.black.map(|below| black_mask_work(&black_field(sm, boost), below, &BlackParams::scaled(g)));
    let dump_dir = dump_dir(opts);
    if dump_dir.is_some() {
        for (name, m) in [("03_lines.png", &line_mask), ("04_blacks.png", &black_mask)] {
            if let Some(m) = m {
                dump(
                    dump_dir,
                    name,
                    &GrayImage::from_raw(work_w, work_h, m.clone()).expect("mask matches the work grid"),
                )?;
            }
        }
    }
    let ink = ink_mask(line_mask, black_mask, work_w, work_h, g.f);
    tm.lap("blacks");
    Ok(ink)
}

/// Write the quantized, separation and final stages of a render made with stages requested.
fn dump_stages(
    rendered: &Rendered,
    pal: &RenderedPalette,
    scfg: &ScreenConfig,
    model: &InkModel,
    dump_dir: Option<&Path>,
) -> Res<()> {
    let index = rendered.index.as_ref().expect("stages were requested");
    let (out_w, out_h) = rendered.image.dimensions();
    let mut q = RgbImage::new(out_w, out_h);
    for (p, &i) in q.pixels_mut().zip(&index.data) {
        p.0 = pal.srgb[i as usize];
    }
    dump(dump_dir, "02_quantized.png", &q)?;
    let field = TintField { index: index.clone(), levels: pal.tints.clone() };
    let cov = screen(&field, scfg);
    for (i, name) in ["05_c.png", "06_m.png", "07_y.png"].iter().enumerate() {
        dump(dump_dir, name, &separation_preview(&cov, i, model))?;
    }
    dump(dump_dir, "08_final.png", &rendered.image)?;
    Ok(())
}

fn run_inner(img: &RgbImage, opts: &Options, want_stages: bool, tm: &mut Timer) -> Res<Inner> {
    let (w_in, h_in) = img.dimensions();
    if w_in == 0 || h_in == 0 {
        return Err(Error::InvalidInput(format!("image has no pixels ({w_in}x{h_in})")));
    }
    let (out_w, out_h) = output_dims(w_in, h_in, opts.scale)?;
    let (pitch, pitch_clamped) = opts.resolve_pitch(out_w);
    if pitch > MAX_PITCH {
        return Err(Error::InvalidOptions(format!("dot pitch {pitch} px is above the maximum of {MAX_PITCH}")));
    }
    let dump_dir = dump_dir(opts);
    if let Some(d) = dump_dir {
        std::fs::create_dir_all(d)?;
    }
    let want_stages = want_stages || dump_dir.is_some();

    let work = to_working(img, opts.work_size);
    let (work_w, work_h) = work.dimensions();
    let g = MaskScale::new(out_w, out_h, work_w);
    tm.lap("working");
    let bilateral = BilateralParams { passes: opts.bilateral_passes, ..Default::default() };
    let sm = smooth_rgb(&work, &bilateral);
    drop(work);
    tm.lap("bilateral");
    if dump_dir.is_some() {
        dump(dump_dir, "01_smoothed.png", &lab_to_rgb(&sm))?;
    }

    let model = InkModel::new(&opts.inks);
    let pal = RenderedPalette::build(&model, &LEVELS, pitch, opts.dot_gain);
    let stats = lightness_stats(&sm);
    let boost = if opts.boost { boost_from_stats(&stats, opts.boost_chroma) } else { Boost::identity() };
    let bw = boost_plane(&sm, &boost);
    let codes = match opts.colors {
        Some(k) => kmeans_codes(&sm, &boost, k, opts.seed, &pal, opts.l_weight),
        None => opts.allowed.clone().unwrap_or_else(all_codes),
    };
    let solid: Vec<u8> = codes.iter().copied().filter(|&c| pal.lab[c as usize][0] >= DARK_CODE_L).collect();
    let palette_codes = if solid.is_empty() || opts.keep_dark_codes { &codes } else { &solid };
    let lut = PaletteLut::with_hue(&pal, opts.l_weight, opts.hue_weight, palette_codes);
    tm.lap("palette");

    let idx = regions(&bw, &lut, opts, &g, out_w, out_h);
    tm.lap("filter");

    let ink = ink_masks(&sm, &boost, stats.p08, opts, &g, (work_w, work_h), tm)?;
    drop(sm);

    let scfg = ScreenConfig::from_params(ScreenParams {
        pitch,
        dot_gain: opts.dot_gain,
        grain: opts.grain,
        misregister: opts.misregister,
        angles: opts.angles,
        seed: opts.seed,
    });
    let rendered = render(
        &RenderInput {
            codes: &idx,
            boosted: &bw,
            lut: &lut,
            pal_lab: &pal.lab,
            l_weight: opts.l_weight,
            ink: ink.as_deref(),
            levels: &pal.tints,
            screen: &scfg,
            model: &model,
            grain: opts.grain,
            seed: opts.seed,
        },
        out_w,
        out_h,
        want_stages,
    );
    tm.lap("render");

    if dump_dir.is_some() {
        dump_stages(&rendered, &pal, &scfg, &model, dump_dir)?;
    }
    Ok(Inner { image: rendered.image, index: rendered.index, alpha: rendered.alpha, pitch, pitch_clamped })
}

/// The warning for a requested dot pitch below [`crate::MIN_PITCH`].
#[must_use]
pub fn pitch_warning() -> String {
    format!("warning: requested dot pitch is below {} px; using {0}", crate::MIN_PITCH)
}

/// Library callers see the clamp warning only with `verbose`; the CLI prints it unconditionally.
pub(crate) fn warn_clamp(opts: &Options, clamped: bool) {
    if clamped && opts.verbose {
        eprintln!("{}", pitch_warning());
    }
}

/// Convert an in-memory image, returning every product. Touches no files unless stage dumps are requested.
///
/// # Errors
/// Invalid options, an empty image, an output beyond [`MAX_OUTPUT_PIXELS`], or a failed stage dump.
pub fn run_image(img: &RgbImage, opts: &Options) -> Res<Output> {
    opts.validate()?;
    let mut tm = Timer::new(opts.verbose);
    let r = run_inner(img, opts, true, &mut tm)?;
    warn_clamp(opts, r.pitch_clamped);
    Ok(Output {
        image: r.image,
        index: r.index.expect("stages were requested"),
        ink_alpha: r.alpha,
        pitch: r.pitch,
        pitch_clamped: r.pitch_clamped,
    })
}

/// Convert an in-memory image to the print only; skips the full-resolution code and alpha buffers.
///
/// # Errors
/// As [`run_image`].
pub fn run_print(img: &RgbImage, opts: &Options) -> Res<RgbImage> {
    opts.validate()?;
    let mut tm = Timer::new(opts.verbose);
    let r = run_inner(img, opts, false, &mut tm)?;
    warn_clamp(opts, r.pitch_clamped);
    Ok(r.image)
}

#[cfg(feature = "io")]
/// Decode `input`, convert, and save to `output`. The output extension is checked before any work is done.
/// Returns a [`Report`]; the clamp warning is left to the caller (see [`pitch_warning`]).
///
/// # Errors
/// Invalid options, an unwritable output name, an unreadable input, or any [`run_image`] error.
pub fn run(input: &Path, output: &Path, opts: &Options) -> Res<Report> {
    opts.validate()?;
    io::check_output(output)?;
    let mut tm = Timer::new(opts.verbose);
    let img = io::decode(input)?;
    tm.lap("decode");
    let r = run_inner(&img, opts, false, &mut tm)?;
    drop(img);
    io::save(&r.image, output, opts.png_compression)?;
    tm.lap("save");
    Ok(Report { pitch: r.pitch, pitch_clamped: r.pitch_clamped })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_size_is_capped() {
        assert_eq!(output_dims(10, 10, 1.5).unwrap(), (15, 15));
        assert_eq!(output_dims(10, 10, 1e-9).unwrap(), (1, 1));
        assert!(output_dims(10, 10, 1e7).is_err());
        assert!(output_dims(10, 10, f32::INFINITY).is_err());
        assert!(output_dims(u32::MAX, u32::MAX, 1.0).is_err());
        assert!(output_dims(16384, 16384, 1.0).is_ok());
    }
}
