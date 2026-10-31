//! Fused full-resolution pass: code upsample, screen, composite and ink alpha, one output row band at a time,
//! straight into the RGB buffer.

use crate::codes::PaletteLut;
use crate::composite::Mixer;
use crate::ink::{InkModel, delta_e};
use crate::noise::{FbmBand, grain_seeds};
use crate::screen::{BAND, ScreenConfig, coverage_row, grain_row, shifted_columns, shifted_row};
use crate::types::{IndexMap, LabPlane};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

/// Everything `render` reads. Region work is on the work grid; the output grid may be larger.
pub struct RenderInput<'a> {
    /// Region codes on the work grid.
    pub codes: &'a IndexMap,
    /// Boosted smoothed Lab on the work grid, for the edge-aware code upsample.
    pub boosted: &'a LabPlane,
    /// Palette lookup that re-snaps pixels on region boundaries.
    pub lut: &'a PaletteLut,
    /// Lab of each code.
    pub pal_lab: &'a [[f32; 3]],
    /// Lightness weight of the colour distance.
    pub l_weight: f32,
    /// Ink (lines and blacks), 0..=255, on the work grid.
    pub ink: Option<&'a [u8]>,
    /// Per-code `[c, m, y]` tints.
    pub levels: &'a [[f32; 3]],
    /// Dot screen.
    pub screen: &'a ScreenConfig,
    /// Ink model.
    pub model: &'a InkModel,
    /// Grain amplitude, for both the tints and the paper.
    pub grain: f32,
    /// Seed for the grain noise; should equal the screen's.
    pub seed: u64,
}

/// Result of [`render`].
pub struct Rendered {
    /// The print.
    pub image: RgbImage,
    /// Upsampled region codes, when stages were requested.
    pub index: Option<IndexMap>,
    /// Ink alpha at output resolution, when stages were requested and there is ink.
    pub alpha: Option<GrayImage>,
}

/// Bilinear footprint of one output row or column in the work grid.
#[derive(Clone, Copy)]
struct Foot {
    i0: u32,
    i1: u32,
    f: f32,
}

fn footprint(o: usize, scale: f32, n: usize) -> Foot {
    let f = ((o as f32 + 0.5) * scale - 0.5).max(0.0).min((n - 1) as f32);
    let i0 = f as usize;
    // A zero-weight neighbour must not count as a disagreeing code (at scale 1 every tap has f = 0).
    let i1 = if f > i0 as f32 { (i0 + 1).min(n - 1) } else { i0 };
    Foot { i0: i0 as u32, i1: i1 as u32, f: f - i0 as f32 }
}

/// Region codes of one output row: take the 2x2 work neighbourhood if it agrees, otherwise re-snap the
/// bilinear-sampled boosted Lab, falling back to the nearest of the four neighbours.
fn code_row(inp: &RenderInput, fx: &[Foot], fy: Foot, out: &mut [u8]) {
    let ww = inp.codes.width as usize;
    let r0 = &inp.codes.data[fy.i0 as usize * ww..][..ww];
    let r1 = &inp.codes.data[fy.i1 as usize * ww..][..ww];
    for (o, f) in out.iter_mut().zip(fx) {
        let (a, b, c, d) = (r0[f.i0 as usize], r0[f.i1 as usize], r1[f.i0 as usize], r1[f.i1 as usize]);
        *o = if a == b && a == c && a == d {
            a
        } else {
            let lab = inp.boosted.bilinear(f.i0 as f32 + f.f, fy.i0 as f32 + fy.f);
            let snapped = inp.lut.lookup(lab);
            if snapped == a || snapped == b || snapped == c || snapped == d {
                snapped
            } else {
                let mut best = a;
                let mut bd = f32::INFINITY;
                for k in [a, b, c, d] {
                    let dd = delta_e(lab, inp.pal_lab[k as usize], inp.l_weight);
                    if dd < bd {
                        bd = dd;
                        best = k;
                    }
                }
                best
            }
        };
    }
}

/// Render the print at `out_w` x `out_h`.
///
/// Work is done per band of 32 output rows plus a margin of `ceil(max |misregistration dy|) + 1` code rows:
///
/// 1. Code upsample (see `code_row`). A work neighbour with zero bilinear weight is not part of the 2x2 set,
///    so at equal grids the index equals `codes` exactly.
/// 2. Per ink, tints are looked up at the misregistered pixel, grained, and turned into 8-bit coverage.
/// 3. Paper grain, the three coverages and the bilinear-sampled key ink are mixed in linear light and
///    converted to sRGB.
///
/// `want_stages == false` skips the full-resolution index and alpha buffers.
///
/// # Panics
/// If `ink` or `boosted` do not match the size of `codes`.
pub fn render(inp: &RenderInput, out_w: u32, out_h: u32, want_stages: bool) -> Rendered {
    let (ow, oh) = (out_w as usize, out_h as usize);
    let (ww, wh) = (inp.codes.width as usize, inp.codes.height as usize);
    assert_eq!((inp.boosted.width as usize, inp.boosted.height as usize), (ww, wh));
    if let Some(ink) = inp.ink {
        assert_eq!(ink.len(), ww * wh, "ink must be on the work grid");
    }
    let fx: Vec<Foot> = (0..ow).map(|x| footprint(x, ww as f32 / ow as f32, ww)).collect();
    let fy: Vec<Foot> = (0..oh).map(|y| footprint(y, wh as f32 / oh as f32, wh)).collect();
    let cfg = inp.screen;
    let grain = inp.grain.max(0.0);
    let seeds = grain_seeds(inp.seed);
    let mixer = Mixer::new(inp.model);
    let lvl: [Vec<f32>; 3] = std::array::from_fn(|k| inp.levels.iter().map(|l| l[k]).collect());
    let cols: Vec<(Vec<u32>, Vec<f32>)> = cfg.inks.iter().map(|s| shifted_columns(s.offset.0, ow)).collect();
    let margin = cfg.inks.iter().map(|s| s.offset.1.abs().ceil() as usize + 1).max().unwrap_or(0);

    let mut image = vec![0u8; ow * oh * 3];
    let mut index = want_stages.then(|| vec![0u8; ow * oh]);
    let mut alpha = (want_stages && inp.ink.is_some()).then(|| vec![0u8; ow * oh]);
    let mut ix_chunks = index.as_deref_mut().map(|v| v.chunks_mut(ow * BAND));
    let mut al_chunks = alpha.as_deref_mut().map(|v| v.chunks_mut(ow * BAND));
    let jobs: Vec<_> = image
        .chunks_mut(ow * 3 * BAND)
        .enumerate()
        .map(|(b, im)| {
            (b, im, ix_chunks.as_mut().and_then(Iterator::next), al_chunks.as_mut().and_then(Iterator::next))
        })
        .collect();

    jobs.into_par_iter().for_each(|(b, im, mut ix_out, mut al_out)| {
        let y0 = b * BAND;
        let rows = im.len() / (ow * 3);
        let ylo = y0.saturating_sub(margin);
        let yhi = (y0 + rows + margin).min(oh);
        let mut codes = vec![0u8; (yhi - ylo) * ow];
        for (y, row) in (ylo..yhi).zip(codes.chunks_mut(ow)) {
            code_row(inp, &fx, fy[y], row);
        }
        let bands: Vec<FbmBand> =
            if grain > 0.0 { seeds.iter().map(|&s| FbmBand::new(y0, rows, s, ow)).collect() } else { Vec::new() };
        let mut noise = [vec![0f32; ow], vec![0f32; ow], vec![0f32; ow], vec![0f32; ow]];
        let mut cov = [vec![0u8; ow], vec![0u8; ow], vec![0u8; ow]];
        let mut tint = vec![0f32; ow];
        let mut arow = vec![0u8; ow];
        for ry in 0..rows {
            let y = y0 + ry;
            if let Some(ix) = ix_out.as_deref_mut() {
                ix[ry * ow..(ry + 1) * ow].copy_from_slice(&codes[(y - ylo) * ow..][..ow]);
            }
            if grain > 0.0 {
                for (n, band) in noise.iter_mut().zip(&bands) {
                    band.row(y, n);
                }
            }
            for k in 0..3 {
                let (sy, ty) = shifted_row(y, cfg.inks[k].offset.1, oh);
                let src = &codes[(ty - ylo) * ow..][..ow];
                let (cmap, sxs) = (&cols[k].0[..ow], &cols[k].1[..ow]);
                let lv = &lvl[k];
                let tv = &mut tint[..ow];
                for x in 0..ow {
                    tv[x] = lv[src[cmap[x] as usize] as usize];
                }
                if grain > 0.0 {
                    grain_row(tv, &noise[k], grain);
                }
                coverage_row(cfg, k, sxs, sy, tv, &mut cov[k][..ow]);
            }
            if let Some(ink) = inp.ink {
                let (f, a) = (fy[y], &mut arow[..ow]);
                let r0 = &ink[f.i0 as usize * ww..][..ww];
                let r1 = &ink[f.i1 as usize * ww..][..ww];
                for (o, g) in a.iter_mut().zip(&fx) {
                    let top = r0[g.i0 as usize] as f32 * (1.0 - g.f) + r0[g.i1 as usize] as f32 * g.f;
                    let bot = r1[g.i0 as usize] as f32 * (1.0 - g.f) + r1[g.i1 as usize] as f32 * g.f;
                    *o = (top * (1.0 - f.f) + bot * f.f + 0.5) as u8;
                }
                if let Some(al) = al_out.as_deref_mut() {
                    al[ry * ow..(ry + 1) * ow].copy_from_slice(a);
                }
            }
            mixer.row([&cov[0], &cov[1], &cov[2]], &noise[3], grain, &arow, &mut im[ry * ow * 3..(ry + 1) * ow * 3]);
        }
    });

    Rendered {
        image: RgbImage::from_raw(out_w, out_h, image).expect("buffer size matches dimensions"),
        index: index.map(|d| IndexMap::from_vec(out_w, out_h, d)),
        alpha: alpha.map(|d| GrayImage::from_raw(out_w, out_h, d).expect("buffer size matches dimensions")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::{RenderedPalette, all_codes};
    use crate::composite::composite;
    use crate::screen::{BENDAY_ANGLES, ScreenParams, screen};
    use crate::types::{InkSet, LEVELS, Plane, TintField};

    struct Fixture {
        model: InkModel,
        pal: RenderedPalette,
        lut: PaletteLut,
    }

    fn fixture() -> Fixture {
        let model = InkModel::new(&InkSet::aged());
        let pal = RenderedPalette::build(&model, &LEVELS, 10.0, 0.0);
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        Fixture { model, pal, lut }
    }

    fn blocky(w: u32, h: u32, seed: u64) -> IndexMap {
        let d = (0..w * h)
            .map(|i| (crate::noise::hash_u64((i % w) as u64 / 5, (i / w) as u64 / 4, seed) % 64) as u8)
            .collect();
        Plane::from_vec(w, h, d)
    }

    fn input<'a>(
        f: &'a Fixture,
        codes: &'a IndexMap,
        boosted: &'a LabPlane,
        cfg: &'a ScreenConfig,
        ink: Option<&'a [u8]>,
    ) -> RenderInput<'a> {
        RenderInput {
            codes,
            boosted,
            lut: &f.lut,
            pal_lab: &f.pal.lab,
            l_weight: 1.5,
            ink,
            levels: &f.pal.tints,
            screen: cfg,
            model: &f.model,
            grain: cfg.grain,
            seed: cfg.seed,
        }
    }

    #[test]
    fn identity_at_equal_grids() {
        let f = fixture();
        let (w, h) = (97u32, 75u32);
        let codes = blocky(w, h, 3);
        let boosted = Plane::new(w, h, [50.0, 0.0, 0.0]);
        for (grain, mis) in [(0.0, 0.0), (0.05, 0.0), (0.0, 0.15), (0.06, 0.2)] {
            let cfg = ScreenConfig::from_params(ScreenParams {
                pitch: 9.0,
                dot_gain: 0.25,
                grain,
                misregister: mis,
                angles: BENDAY_ANGLES,
                seed: 11,
            });
            let r = render(&input(&f, &codes, &boosted, &cfg, None), w, h, true);
            assert_eq!(r.index.as_ref().unwrap(), &codes);
            assert!(r.alpha.is_none());
            let field = TintField { index: codes.clone(), levels: f.pal.tints.clone() };
            let want = composite(&screen(&field, &cfg), None, &f.model, grain, 11);
            for (a, b) in r.image.as_raw().iter().zip(want.as_raw()) {
                assert!((*a as i32 - *b as i32).abs() <= 1, "grain {grain} mis {mis}");
            }
            let lean = render(&input(&f, &codes, &boosted, &cfg, None), w, h, false);
            assert!(lean.index.is_none() && lean.image == r.image);
        }
    }

    #[test]
    fn identity_with_ink_alpha() {
        let f = fixture();
        let (w, h) = (64u32, 40u32);
        let codes = blocky(w, h, 8);
        let boosted = Plane::new(w, h, [50.0, 0.0, 0.0]);
        let ink: Vec<u8> = (0..w * h).map(|i| (i * 37 % 256) as u8).collect();
        let cfg = ScreenConfig::from_params(ScreenParams {
            pitch: 8.0,
            dot_gain: 0.0,
            grain: 0.03,
            misregister: 0.1,
            angles: BENDAY_ANGLES,
            seed: 2,
        });
        let r = render(&input(&f, &codes, &boosted, &cfg, Some(&ink)), w, h, true);
        assert_eq!(r.alpha.as_ref().unwrap().as_raw(), &ink);
        let field = TintField { index: codes.clone(), levels: f.pal.tints.clone() };
        let want = composite(&screen(&field, &cfg), r.alpha.as_ref(), &f.model, 0.03, 2);
        for (a, b) in r.image.as_raw().iter().zip(want.as_raw()) {
            assert!((*a as i32 - *b as i32).abs() <= 1);
        }
    }

    #[test]
    fn upsampled_edge_lands_near_the_true_edge() {
        let f = fixture();
        let (a, b) = (0u8, 63u8);
        let (ww, wh) = (20u32, 6u32);
        let codes = Plane::from_vec(ww, wh, (0..ww * wh).map(|i| if i % ww < 10 { a } else { b }).collect());
        let boosted = Plane::from_vec(
            ww,
            wh,
            (0..ww * wh).map(|i| f.pal.lab[if i % ww < 10 { a } else { b } as usize]).collect(),
        );
        let cfg = ScreenConfig::from_params(ScreenParams {
            pitch: 9.0,
            dot_gain: 0.0,
            grain: 0.0,
            misregister: 0.0,
            angles: BENDAY_ANGLES,
            seed: 1,
        });
        let (ow, oh) = (60u32, 18u32);
        let r = render(&input(&f, &codes, &boosted, &cfg, None), ow, oh, true);
        let idx = r.index.unwrap();
        // Work boundary between columns 9 and 10 sits at output x = 30.
        for y in 0..oh {
            let first_b = (0..ow).find(|&x| idx.get(x, y) == b).unwrap();
            assert!((29..=31).contains(&first_b), "row {y}: {first_b}");
            assert!((0..first_b).all(|x| idx.get(x, y) == a));
            assert!((first_b..ow).all(|x| idx.get(x, y) == b));
        }
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let f = fixture();
        let codes = blocky(50, 40, 1);
        let boosted = Plane::new(50, 40, [50.0, 0.0, 0.0]);
        let cfg = ScreenConfig::from_params(ScreenParams {
            pitch: 9.0,
            dot_gain: 0.25,
            grain: 0.05,
            misregister: 0.15,
            angles: BENDAY_ANGLES,
            seed: 7,
        });
        let ink = vec![90u8; 50 * 40];
        let run = || render(&input(&f, &codes, &boosted, &cfg, Some(&ink)), 131, 97, false).image.into_raw();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(run);
        assert_eq!(one, run());
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_12mp() {
        let f = fixture();
        let (ww, wh, ow, oh) = (2016u32, 1512u32, 4032u32, 3024u32);
        let codes = blocky(ww, wh, 5);
        let boosted = Plane::new(ww, wh, [50.0, 0.0, 0.0]);
        let ink: Vec<u8> = (0..ww * wh).map(|i| if i % 97 < 3 { 255 } else { 0 }).collect();
        for (name, grain, mis, with_ink) in
            [("comic", 0.04, 0.15, true), ("pop", 0.0, 0.0, true), ("comic-nolines", 0.04, 0.15, false)]
        {
            let cfg = ScreenConfig::from_params(ScreenParams {
                pitch: 26.9,
                dot_gain: 0.25,
                grain,
                misregister: mis,
                angles: BENDAY_ANGLES,
                seed: 1,
            });
            let inp = input(&f, &codes, &boosted, &cfg, with_ink.then_some(&ink[..]));
            let (w, c) = crate::testutil::best_of(5, || render(&inp, ow, oh, false));
            eprintln!("render {name}: wall {w:.1} ms, cpu {c:.1} ms");
        }
    }
}
