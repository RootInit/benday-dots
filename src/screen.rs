use crate::noise::{FbmBand, SALT_MISREG, SALT_PHASE_U, SALT_PHASE_V, grain_seeds, hash01};
use crate::types::{Coverage, TintField};
use rayon::prelude::*;
use std::f32::consts::PI;

/// Rows per parallel band (also the span an `FbmBand` shares lattice rows over).
pub(crate) const BAND: usize = 32;

/// One ink's screen: angle, lattice phase and misregistration offset.
#[derive(Clone, Copy, Debug)]
pub struct InkScreen {
    /// Lattice rotation in degrees.
    pub angle_deg: f32,
    /// Lattice phase in cells.
    pub phase: (f32, f32),
    /// Misregistration shift in pixels.
    pub offset: (f32, f32),
}

/// Scalar parameters of a [`ScreenConfig`]; a struct so the three `f32`s cannot be swapped.
#[derive(Clone, Copy, Debug)]
pub struct ScreenParams {
    /// Dot lattice pitch in pixels.
    pub pitch: f32,
    /// Dot gain in pixels.
    pub dot_gain: f32,
    /// Tint grain amplitude.
    pub grain: f32,
    /// Misregistration magnitude as a fraction of the pitch.
    pub misregister: f32,
    /// Lattice angles for C, M, Y in degrees.
    pub angles: [f32; 3],
    /// Seed for phases, misregistration directions and grain.
    pub seed: u64,
}

/// Halftone screen for the three inks.
#[derive(Clone, Debug)]
pub struct ScreenConfig {
    /// Dot lattice pitch in pixels.
    pub pitch: f32,
    /// Dot gain in pixels.
    pub dot_gain: f32,
    /// Tint grain amplitude.
    pub grain: f32,
    /// Seed for phases, misregistration directions and grain.
    pub seed: u64,
    /// Per-ink screens, C, M, Y.
    pub inks: [InkScreen; 3],
}

impl ScreenConfig {
    /// Phase and misregister direction come from hash(seed, ink).
    pub fn from_params(p: ScreenParams) -> Self {
        let ink = |i: usize| {
            let ii = i as i64;
            let phi = 2.0 * PI * hash01(ii, SALT_MISREG, p.seed);
            let mag = p.misregister * p.pitch;
            InkScreen {
                angle_deg: p.angles[i],
                phase: (hash01(ii, SALT_PHASE_U, p.seed), hash01(ii, SALT_PHASE_V, p.seed)),
                offset: (mag * phi.cos(), mag * phi.sin()),
            }
        };
        ScreenConfig {
            pitch: p.pitch,
            dot_gain: p.dot_gain,
            grain: p.grain,
            seed: p.seed,
            inks: [ink(0), ink(1), ink(2)],
        }
    }
}

/// Default angles for C, M, Y.
pub const BENDAY_ANGLES: [f32; 3] = [15.0, 45.0, 75.0];

/// Per-ink lattice with trigonometry hoisted out of the per-pixel loop.
#[derive(Clone, Copy)]
struct Lattice {
    cos: f32,
    sin: f32,
    ox: f32,
    oy: f32,
}

impl Lattice {
    fn new(s: &InkScreen) -> Self {
        let th = s.angle_deg.to_radians();
        Lattice { cos: th.cos(), sin: th.sin(), ox: s.phase.0, oy: s.phase.1 }
    }
}

/// floor() without a libm call on baseline x86-64 (no SSE4.1); exact for |a| < 2^22.
#[inline(always)]
pub(crate) fn ffloor(a: f32) -> f32 {
    // Round-to-nearest via the 1.5*2^23 trick keeps it pure float arithmetic, so it vectorises on SSE2.
    const M: f32 = 12_582_912.0;
    let r = (a + M) - M;
    if r > a { r - 1.0 } else { r }
}

/// Scalar coverage: the oracle for the vectorised `coverage_row`.
#[cfg(test)]
fn coverage_lat(cfg: &ScreenConfig, l: &Lattice, px: f32, py: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let p = cfg.pitch;
    let g = cfg.dot_gain;
    let u = (px * l.cos + py * l.sin) / p + l.ox;
    let v = (-px * l.sin + py * l.cos) / p + l.oy;
    let frac = |a: f32| a - a.floor() - 0.5;
    let half = p * 0.5;
    if t <= 0.5 {
        let (fu, fv) = (frac(u), frac(v));
        let d = p * (fu * fu + fv * fv).sqrt();
        let r = (p * (t / PI).sqrt() + g).clamp(0.0, half);
        (r - d + 0.5).clamp(0.0, 1.0)
    } else {
        let (fu, fv) = (frac(u + 0.5), frac(v + 0.5));
        let d = p * (fu * fu + fv * fv).sqrt();
        let rh = (p * ((1.0 - t) / PI).sqrt() - g).clamp(0.0, half);
        1.0 - (rh - d + 0.5).clamp(0.0, 1.0)
    }
}

/// Pure analytic coverage of ink `ink` at screen-space point (px, py) for tint t. No grain, no offset.
#[cfg(test)]
pub(crate) fn coverage_at(cfg: &ScreenConfig, ink: usize, px: f32, py: f32, t: f32) -> f32 {
    coverage_lat(cfg, &Lattice::new(&cfg.inks[ink]), px, py, t)
}

/// One ink's coverage (0..=255) for a row: `sx[x]` is the screen-space x of pixel x, `sy` the row's, and
/// `t[x]` the tint already looked up and grained. The loop is branch-free so it vectorises.
#[allow(clippy::manual_clamp)] // clamp's min <= max assert blocks vectorisation
pub(crate) fn coverage_row(cfg: &ScreenConfig, ink: usize, sx: &[f32], sy: f32, t: &[f32], out: &mut [u8]) {
    let n = out.len();
    let (sx, t) = (&sx[..n], &t[..n]);
    let lat = Lattice::new(&cfg.inks[ink]);
    let (ip, gain) = (cfg.pitch, cfg.dot_gain);
    let half = ip * 0.5;
    let (us, vs) = (sy * lat.sin, sy * lat.cos);
    for x in 0..n {
        let tt = t[x];
        let u = (sx[x] * lat.cos + us) / ip + lat.ox;
        let v = (-sx[x] * lat.sin + vs) / ip + lat.oy;
        let lo = tt <= 0.5;
        let (uu, vv) = if lo { (u, v) } else { (u + 0.5, v + 0.5) };
        let fu = uu - ffloor(uu) - 0.5;
        let fv = vv - ffloor(vv) - 0.5;
        let d = ip * (fu * fu + fv * fv).sqrt();
        let rr = if lo {
            (ip * (tt / PI).sqrt() + gain).max(0.0).min(half)
        } else {
            (ip * ((1.0 - tt) / PI).sqrt() - gain).max(0.0).min(half)
        };
        let c = (rr - d + 0.5).max(0.0).min(1.0);
        let c = if lo { c } else { 1.0 - c };
        // SAFETY: c is in [0, 1].
        let q: u8 = unsafe { (c * 255.0 + 0.5).to_int_unchecked::<i32>() } as u8;
        out[x] = if tt <= 0.0 {
            0
        } else if tt >= 1.0 {
            255
        } else {
            q
        };
    }
}

/// Apply paper-relative grain to a row of tints; exact endpoints are left alone.
#[allow(clippy::manual_clamp)] // see coverage_row
pub(crate) fn grain_row(t: &mut [f32], noise: &[f32], grain: f32) {
    let n = t.len();
    let noise = &noise[..n];
    for x in 0..n {
        let v = t[x];
        let g = (v * (1.0 + grain * (2.0 * noise[x] - 1.0))).max(0.0).min(1.0);
        t[x] = if v <= 0.0 || v >= 1.0 { v } else { g };
    }
}

/// Per-ink misregistration along x: source column of each output pixel, and its screen-space x.
pub(crate) fn shifted_columns(dx: f32, w: usize) -> (Vec<u32>, Vec<f32>) {
    let sx: Vec<f32> = (0..w).map(|x| x as f32 + 0.5 - dx).collect();
    let cols = sx.iter().map(|&s| (ffloor(s) as i64).clamp(0, w as i64 - 1) as u32).collect();
    (cols, sx)
}

/// Screen-space y of output row `y` for an ink offset `dy`, and the source row it reads.
#[inline]
pub(crate) fn shifted_row(y: usize, dy: f32, h: usize) -> (f32, usize) {
    let sy = y as f32 + 0.5 - dy;
    (sy, (ffloor(sy) as i64).clamp(0, h as i64 - 1) as usize)
}

/// Full screen: misregistration offset, per-pixel tint lookup, grain. rayon over row bands. Deterministic.
pub fn screen(field: &TintField, cfg: &ScreenConfig) -> Coverage {
    let (w, h) = (field.width(), field.height());
    let (wu, hu) = (w as usize, h as usize);
    let seeds = grain_seeds(cfg.seed);
    let cols: Vec<(Vec<u32>, Vec<f32>)> = cfg.inks.iter().map(|s| shifted_columns(s.offset.0, wu)).collect();
    let mut planes = [vec![0u8; wu * hu], vec![0u8; wu * hu], vec![0u8; wu * hu]];
    let [p0, p1, p2] = &mut planes;
    p0.par_chunks_mut(wu * BAND)
        .zip(p1.par_chunks_mut(wu * BAND))
        .zip(p2.par_chunks_mut(wu * BAND))
        .enumerate()
        .for_each(|(b, ((b0, b1), b2))| {
            let y0 = b * BAND;
            let rows = b0.len() / wu;
            let bands: Vec<FbmBand> = if cfg.grain > 0.0 {
                seeds[..3].iter().map(|&s| FbmBand::new(y0, rows, s, wu)).collect()
            } else {
                Vec::new()
            };
            let mut noise = vec![0f32; wu];
            let mut tv = vec![0f32; wu];
            for (ink, band) in [b0, b1, b2].into_iter().enumerate() {
                let (cmap, sxs) = &cols[ink];
                for ry in 0..rows {
                    let (sy, ty) = shifted_row(y0 + ry, cfg.inks[ink].offset.1, hu);
                    for (v, &c) in tv.iter_mut().zip(cmap) {
                        *v = field.tint(c, ty as u32, ink);
                    }
                    if cfg.grain > 0.0 {
                        bands[ink].row(y0 + ry, &mut noise);
                        grain_row(&mut tv, &noise, cfg.grain);
                    }
                    coverage_row(cfg, ink, sxs, sy, &tv, &mut band[ry * wu..(ry + 1) * wu]);
                }
            }
        });
    Coverage { width: w, height: h, planes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::composite;
    use crate::ink::{InkModel, effective_coverage};
    use crate::types::{Code, IndexMap, InkSet, LEVELS};

    fn cfg(p: f32, gain: f32, theta: f32) -> ScreenConfig {
        let mut c = ScreenConfig::from_params(ScreenParams {
            pitch: p,
            dot_gain: gain,
            grain: 0.0,
            misregister: 0.0,
            angles: [theta; 3],
            seed: 1,
        });
        for s in &mut c.inks {
            s.phase = (0.0, 0.0);
        }
        c
    }

    fn tile_mean(c: &ScreenConfig, t: f32) -> f32 {
        let mut sum = 0.0f64;
        for y in 0..256 {
            for x in 0..256 {
                sum += coverage_at(c, 0, x as f32 + 0.5, y as f32 + 0.5, t) as f64;
            }
        }
        (sum / 65536.0) as f32
    }

    fn steps() -> impl Iterator<Item = f32> {
        (0..=20).map(|i| i as f32 * 0.05)
    }

    #[test]
    fn mean_coverage_matches_tint() {
        for (p, tol) in [(6.0, 0.02), (10.0, 0.01), (16.0, 0.01)] {
            let c = cfg(p, 0.0, 15.0);
            for t in steps() {
                let m = tile_mean(&c, t);
                assert!((m - t).abs() < tol, "p={p} t={t} mean={m}");
            }
        }
    }

    #[test]
    fn mean_is_monotone() {
        let c = cfg(10.0, 0.0, 15.0);
        let mut prev = -1.0;
        for t in steps() {
            let m = tile_mean(&c, t);
            assert!(m >= prev - 1e-4, "t={t}");
            prev = m;
        }
    }

    #[test]
    fn endpoints_exact_for_any_gain() {
        for g in [-1.0, 0.0, 0.5, 3.0] {
            let c = cfg(8.0, g, 33.0);
            for y in 0..40 {
                for x in 0..40 {
                    assert_eq!(coverage_at(&c, 0, x as f32 + 0.5, y as f32 + 0.5, 0.0), 0.0);
                    assert_eq!(coverage_at(&c, 0, x as f32 + 0.5, y as f32 + 0.5, 1.0), 1.0);
                }
            }
        }
    }

    #[test]
    fn angle_invariance() {
        for t in [0.2, 0.5, 0.8] {
            let means: Vec<f32> = [0.0, 15.0, 45.0, 75.0]
                .iter()
                .map(|&a| {
                    let mut c = cfg(10.0, 0.0, a);
                    c.inks[0].phase = (0.137, 0.291);
                    tile_mean(&c, t)
                })
                .collect();
            for m in &means {
                assert!((m - t).abs() < 0.015, "t={t} {means:?}");
            }
            let (a, b) = (cfg(10.0, 0.0, 15.0), cfg(10.0, 0.0, 105.0));
            for y in 0..64 {
                for x in 0..64 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let (va, vb) = (coverage_at(&a, 0, px, py, t), coverage_at(&b, 0, px, py, t));
                    assert!((va - vb).abs() < 1e-2, "({x},{y}) {va} {vb}");
                }
            }
        }
    }

    #[test]
    fn matches_supersampled_oracle() {
        for (t, p, theta) in [(0.25, 10.0f32, 15.0f32), (0.5, 10.0, 45.0), (0.75, 12.0, 75.0)] {
            let c = cfg(p, 0.0, theta);
            let (cs, sn) = (theta.to_radians().cos(), theta.to_radians().sin());
            let hard = |px: f32, py: f32| {
                let u = (px * cs + py * sn) / p;
                let v = (-px * sn + py * cs) / p;
                let f = |a: f32| a - a.floor() - 0.5;
                if t <= 0.5 {
                    let d = p * (f(u).powi(2) + f(v).powi(2)).sqrt();
                    (d <= p * (t / PI).sqrt()) as u8 as f32
                } else {
                    let d = p * (f(u + 0.5).powi(2) + f(v + 0.5).powi(2)).sqrt();
                    (d > p * ((1.0 - t) / PI).sqrt()) as u8 as f32
                }
            };
            let mut err = 0.0f64;
            for y in 0..48 {
                for x in 0..48 {
                    let mut ss = 0.0;
                    for j in 0..16 {
                        for i in 0..16 {
                            ss += hard(x as f32 + (i as f32 + 0.5) / 16.0, y as f32 + (j as f32 + 0.5) / 16.0);
                        }
                    }
                    let v = coverage_at(&c, 0, x as f32 + 0.5, y as f32 + 0.5, t);
                    err += (v - ss / 256.0).abs() as f64;
                }
            }
            let mean = err / (48.0 * 48.0);
            assert!(mean < 0.05, "t={t} p={p} theta={theta} err={mean}");
        }
    }

    #[test]
    fn dot_gain_matches_effective_coverage() {
        let c = cfg(10.0, 0.5, 15.0);
        for t in LEVELS {
            let m = tile_mean(&c, t);
            assert!((m - effective_coverage(t, 10.0, 0.5)).abs() < 0.01, "t={t} mean={m}");
        }
    }

    fn pattern_field(w: u32, h: u32) -> TintField {
        let data: Vec<u8> = (0..w * h)
            .map(|i| (crate::noise::hash_u64((i % w) as u64 / 8, (i / w) as u64 / 8, 5) % 64) as u8)
            .collect();
        let levels = (0..64).map(|i| Code::from_index(i).tints(&LEVELS)).collect();
        TintField { index: IndexMap::from_vec(w, h, data), levels }
    }

    #[test]
    fn screen_matches_coverage_at_and_offset_magnitude() {
        let field = pattern_field(96, 96);
        let c = ScreenConfig::from_params(ScreenParams {
            pitch: 10.0,
            dot_gain: 0.0,
            grain: 0.0,
            misregister: 0.0,
            angles: BENDAY_ANGLES,
            seed: 3,
        });
        let cov = screen(&field, &c);
        for ink in 0..3 {
            for y in 0..96u32 {
                for x in 0..96u32 {
                    let t = field.tint(x, y, ink);
                    let want = coverage_at(&c, ink, x as f32 + 0.5, y as f32 + 0.5, t);
                    let got = cov.planes[ink][(y * 96 + x) as usize] as f32 / 255.0;
                    assert!((got - want).abs() <= 1.0 / 255.0 + 1e-6, "ink {ink} ({x},{y})");
                }
            }
        }
        let m = ScreenConfig::from_params(ScreenParams {
            pitch: 10.0,
            dot_gain: 0.0,
            grain: 0.0,
            misregister: 0.2,
            angles: BENDAY_ANGLES,
            seed: 3,
        });
        for s in &m.inks {
            assert!((s.offset.0.hypot(s.offset.1) - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn deterministic_across_threads_runs_and_seeds() {
        let field = pattern_field(128, 128);
        let model = InkModel::new(&InkSet::aged());
        let render = |seed: u64| {
            let c = ScreenConfig::from_params(ScreenParams {
                pitch: 9.0,
                dot_gain: 0.25,
                grain: 0.05,
                misregister: 0.15,
                angles: BENDAY_ANGLES,
                seed,
            });
            composite(&screen(&field, &c), None, &model, 0.05, seed).into_raw()
        };
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| render(7));
        assert_eq!(one, render(7));
        assert_eq!(render(7), render(7));
        assert_ne!(render(7), render(8));
    }

    #[test]
    fn coverage_row_matches_coverage_at() {
        let w = 200;
        for gain in [-0.5f32, 0.0, 0.7] {
            let mut c = ScreenConfig::from_params(ScreenParams {
                pitch: 9.0,
                dot_gain: gain,
                grain: 0.0,
                misregister: 0.0,
                angles: BENDAY_ANGLES,
                seed: 4,
            });
            c.inks[1].phase = (0.3, 0.6);
            let ts: Vec<f32> = (0..w)
                .map(|i| {
                    [0.0, 1.0, 0.5, 0.5001, 0.25, 0.75][i % 6] * (i % 5 != 4) as u8 as f32
                        + (i % 5 == 4) as u8 as f32 * i as f32 / w as f32
                })
                .collect();
            let sx: Vec<f32> = (0..w).map(|x| x as f32 + 0.5 - 0.37).collect();
            for ink in 0..3 {
                let sy = 17.5 + ink as f32;
                let mut out = vec![0u8; w];
                coverage_row(&c, ink, &sx, sy, &ts, &mut out);
                for x in 0..w {
                    let want = coverage_at(&c, ink, sx[x], sy, ts[x]);
                    assert!((out[x] as f32 / 255.0 - want).abs() <= 1.0 / 255.0 + 1e-6, "g={gain} x={x} t={}", ts[x]);
                }
            }
        }
    }
}
