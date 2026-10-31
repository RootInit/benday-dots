//! Edge-preserving smoothing, the working-grid resize and the sRGB/Lab conversions around them.

use crate::ink::{lab_to_srgb8, srgb8_to_lin};
use crate::types::LabPlane;
use rayon::prelude::*;

/// Parameters of [`bilateral_lab`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilateralParams {
    /// Spatial Gaussian sigma in pixels.
    pub sigma_s: f32,
    /// Window radius in pixels; the window is `2 * radius + 1` wide.
    pub radius: u32,
    /// Range sigma in Lab units; taps further than 3 sigma are ignored.
    pub sigma_r: f32,
    /// Number of times the (horizontal, vertical) filter pair is applied.
    pub passes: u32,
}

impl Default for BilateralParams {
    fn default() -> Self {
        BilateralParams { sigma_s: 3.0, radius: 5, sigma_r: 6.0, passes: 3 }
    }
}

/// Triangle resize; returns a clone if the long edge is already <= work_size.
pub fn to_working(img: &image::RgbImage, work_size: u32) -> image::RgbImage {
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= work_size {
        return img.clone();
    }
    let f = work_size as f64 / long as f64;
    let nw = ((w as f64 * f).round() as u32).max(1);
    let nh = ((h as f64 * f).round() as u32).max(1);
    resize_triangle(img, nw, nh)
}

/// Triangle filter taps for one axis: first source index and normalised weights per output index.
fn tri_weights(src: u32, dst: u32) -> Vec<(u32, Vec<f32>)> {
    let ratio = src as f32 / dst as f32;
    let sratio = ratio.max(1.0);
    let support = sratio;
    (0..dst)
        .map(|o| {
            let c = (o as f32 + 0.5) * ratio;
            let left = ((c - support).floor() as i64).clamp(0, src as i64 - 1) as u32;
            let right = ((c + support).ceil() as i64).clamp(left as i64 + 1, src as i64) as u32;
            let c = c - 0.5;
            let mut ws: Vec<f32> = (left..right)
                .map(|i| {
                    let x = ((i as f32 - c) / sratio).abs();
                    if x < 1.0 { 1.0 - x } else { 0.0 }
                })
                .collect();
            let s: f32 = ws.iter().sum();
            for w in &mut ws {
                *w /= s;
            }
            (left, ws)
        })
        .collect()
}

/// `image::imageops::resize(.., FilterType::Triangle)` arithmetic (vertical pass to f32, then horizontal with
/// round-to-nearest), parallel over output rows. Within 1 LSB of image's result.
pub fn resize_triangle(img: &image::RgbImage, w: u32, h: u32) -> image::RgbImage {
    let (sw, sh) = img.dimensions();
    if (sw, sh) == (w, h) || w == 0 || h == 0 || sw == 0 || sh == 0 {
        return if w == 0 || h == 0 { image::RgbImage::new(w, h) } else { img.clone() };
    }
    let vw = tri_weights(sh, h);
    let hw = tri_weights(sw, w);
    let raw = img.as_raw();
    let row_len = sw as usize * 3;
    let mut out = vec![0u8; w as usize * h as usize * 3];
    out.par_chunks_mut(w as usize * 3).enumerate().for_each_init(
        || vec![0f32; row_len],
        |tmp, (y, orow)| {
            let (top, ws) = &vw[y];
            tmp.fill(0.0);
            for (i, &wt) in ws.iter().enumerate() {
                let start = (*top as usize + i) * row_len;
                let srow = &raw[start..start + row_len];
                for (t, &s) in tmp.iter_mut().zip(srow) {
                    *t += s as f32 * wt;
                }
            }
            for (px, (left, ws)) in orow.as_chunks_mut::<3>().0.iter_mut().zip(&hw) {
                let mut acc = [0f32; 3];
                let base = *left as usize * 3;
                for (src, &wt) in tmp[base..base + ws.len() * 3].as_chunks::<3>().0.iter().zip(ws) {
                    acc[0] += src[0] * wt;
                    acc[1] += src[1] * wt;
                    acc[2] += src[2] * wt;
                }
                for (o, a) in px.iter_mut().zip(acc) {
                    // clamp() would block vectorisation.
                    #[allow(clippy::manual_clamp)]
                    let v = a.max(0.0).min(255.0);
                    *o = (v + 0.5) as u8;
                }
            }
        },
    );
    image::RgbImage::from_raw(w, h, out).expect("buffer matches dims")
}

/// Pixels handled together by the SIMD kernels.
const LANES: usize = 8;
type V = [f32; LANES];

/// Output rows per task. Each task also filters `radius` rows above and below horizontally, so taller
/// tasks waste less on that halo but balance worse across the P and E cores.
const V_ROWS: usize = 64;
/// Output samples per task and per scratch row; a multiple of [`LANES`].
const TILE: usize = 256;

#[inline(always)]
unsafe fn load(p: *const f32) -> V {
    // SAFETY: caller guarantees LANES readable floats at `p`.
    unsafe { p.cast::<V>().read_unaligned() }
}

#[inline(always)]
unsafe fn store(p: *mut f32, v: V) {
    // SAFETY: caller guarantees LANES writable floats at `p`.
    unsafe { p.cast::<V>().write_unaligned(v) }
}

/// Coefficients of the degree-4 minimax polynomial for 2^f on (-1, 0], relative error 3e-6.
const EXP2_POLY: [f32; 5] = [0.999_997_4, 0.693_030_05, 0.239_340_75, 0.053_073_91, 0.006_767_128];

/// Constants shared by every row of one filter call.
struct Kern {
    /// Spatial Gaussian per tap offset, `2 * radius + 1` entries.
    sp: Vec<f32>,
    r: usize,
    max_d2: f32,
    /// `-log2(e) / (2 sigma_r^2)`: squared Lab distance to the base-2 exponent.
    inv2: f32,
}

impl Kern {
    fn new(p: &BilateralParams) -> Self {
        let r = p.radius as usize;
        let sp = (0..=2 * r)
            .map(|k| {
                let d = k as f32 - r as f32;
                (-(d * d) / (2.0 * p.sigma_s * p.sigma_s)).exp()
            })
            .collect();
        Kern {
            sp,
            r,
            max_d2: 9.0 * p.sigma_r * p.sigma_r,
            inv2: -std::f32::consts::LOG2_E / (2.0 * p.sigma_r * p.sigma_r),
        }
    }
}

/// Portable kernel: filters `blocks * LANES` pixels along `taps` (`taps[t][ch]` points at the first pixel's
/// plane-`ch` sample of tap `t`), writing to `out[ch]`. The centre tap has weight exactly 1.
///
/// # Safety
/// Every tap pointer must be readable and every `out[ch]` writable for `blocks * LANES` floats.
unsafe fn filter_portable(k: &Kern, taps: &[[*const f32; 3]], out: [*mut f32; 3], blocks: usize) {
    let r = k.r;
    for b in 0..blocks {
        let x = b * LANES;
        // SAFETY: forwarded from the caller.
        unsafe {
            let c = [load(taps[r][0].add(x)), load(taps[r][1].add(x)), load(taps[r][2].add(x))];
            let mut acc = [c[0], c[1], c[2], [1.0; LANES]];
            for (t_idx, q) in taps.iter().enumerate() {
                if t_idx == r {
                    continue;
                }
                let q = [load(q[0].add(x)), load(q[1].add(x)), load(q[2].add(x))];
                // One simple loop per stage so each vectorises on its own.
                let mut dd = [0f32; LANES];
                for j in 0..LANES {
                    let (d0, d1, d2) = (q[0][j] - c[0][j], q[1][j] - c[1][j], q[2][j] - c[2][j]);
                    dd[j] = d0 * d0 + d1 * d1 + d2 * d2;
                }
                let mut t = [0f32; LANES];
                for j in 0..LANES {
                    t[j] = (dd[j] * k.inv2).max(-30.0);
                }
                let mut ti = [0i32; LANES];
                for j in 0..LANES {
                    // t is in [-30, 0], so the truncation fits an i32.
                    ti[j] = t[j].to_int_unchecked();
                }
                let mut wt = [0f32; LANES];
                for j in 0..LANES {
                    let f = t[j] - ti[j] as f32;
                    let p =
                        f * (f * (f * (f * EXP2_POLY[4] + EXP2_POLY[3]) + EXP2_POLY[2]) + EXP2_POLY[1]) + EXP2_POLY[0];
                    let e = f32::from_bits(((ti[j] + 127) << 23) as u32) * p;
                    wt[j] = if dd[j] < k.max_d2 { k.sp[t_idx] * e } else { 0.0 };
                }
                for j in 0..LANES {
                    acc[0][j] += wt[j] * q[0][j];
                    acc[1][j] += wt[j] * q[1][j];
                    acc[2][j] += wt[j] * q[2][j];
                    acc[3][j] += wt[j];
                }
            }
            for ch in 0..3 {
                let mut o = [0f32; LANES];
                for j in 0..LANES {
                    o[j] = acc[ch][j] / acc[3][j];
                }
                store(out[ch].add(x), o);
            }
        }
    }
}

fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    return std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Cube root for x >= 0: exponent-halving initial guess, then two Halley steps (cubic convergence).
fn cbrt_v(x: V) -> V {
    let mut y = [0f32; LANES];
    for j in 0..LANES {
        let guess = (x[j].to_bits() as i32 as f32 * (1.0 / 3.0)) as i32 + CBRT_MAGIC;
        y[j] = f32::from_bits(guess as u32);
    }
    for _ in 0..2 {
        for j in 0..LANES {
            let y3 = y[j] * y[j] * y[j];
            y[j] = y[j] * (y3 + 2.0 * x[j]) / (2.0 * y3 + x[j]);
        }
    }
    y
}

const CBRT_MAGIC: i32 = 0x2A51_2CE0;
/// Below this the Lab transfer function is linear.
const LAB_EPS: f32 = 216.0 / 24389.0;
const WHITE: [f32; 3] = [0.950_47, 1.0, 1.088_83];

fn lab_f(t: V) -> V {
    let c = cbrt_v(t);
    std::array::from_fn(|j| if t[j] > LAB_EPS { c[j] } else { (24389.0 / 27.0 * t[j] + 16.0) / 116.0 })
}

/// Linear sRGB to Lab for `LANES` pixels.
fn lab_math(lin: &[V; 3]) -> [V; 3] {
    let mut t = [[0f32; LANES]; 3];
    for j in 0..LANES {
        let (r, g, b) = (lin[0][j], lin[1][j], lin[2][j]);
        t[0][j] = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / WHITE[0];
        t[1][j] = (0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b) / WHITE[1];
        t[2][j] = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / WHITE[2];
    }
    let (fx, fy, fz) = (lab_f(t[0]), lab_f(t[1]), lab_f(t[2]));
    [
        std::array::from_fn(|j| 116.0 * fy[j] - 16.0),
        std::array::from_fn(|j| 500.0 * (fx[j] - fy[j])),
        std::array::from_fn(|j| 200.0 * (fy[j] - fz[j])),
    ]
}

/// Converts `blocks * LANES` pixels to Lab; pixels beyond the end of `rgb` repeat the last one.
#[inline(always)]
unsafe fn rgb_row_body<const SIMD: bool>(rgb: &[u8], lut: &[f32; 256], out: [*mut f32; 3], blocks: usize) {
    let n = rgb.len() / 3;
    for b in 0..blocks {
        let mut lin = [[0f32; LANES]; 3];
        for (j, px) in (0..LANES).map(|j| &rgb[(b * LANES + j).min(n - 1) * 3..][..3]).enumerate() {
            lin[0][j] = lut[px[0] as usize];
            lin[1][j] = lut[px[1] as usize];
            lin[2][j] = lut[px[2] as usize];
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `SIMD` is only instantiated inside a function with AVX2+FMA enabled.
        let o = if SIMD { unsafe { simd::lab_math(&lin) } } else { lab_math(&lin) };
        #[cfg(not(target_arch = "x86_64"))]
        let o = lab_math(&lin);
        // SAFETY: the caller provides room for `blocks * LANES` floats per plane.
        unsafe {
            for ch in 0..3 {
                store(out[ch].add(b * LANES), o[ch]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rgb_row_avx2(rgb: &[u8], lut: &[f32; 256], out: [*mut f32; 3], blocks: usize) {
    // SAFETY: forwarded from the caller.
    unsafe { rgb_row_body::<true>(rgb, lut, out, blocks) }
}

/// # Safety
/// Each `out[ch]` must be writable for `blocks * LANES` floats; `rgb` must hold at least one pixel.
unsafe fn rgb_row(avx2: bool, rgb: &[u8], lut: &[f32; 256], out: [*mut f32; 3], blocks: usize) {
    debug_assert!(!rgb.is_empty());
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: `avx2` is only true after runtime detection.
        return unsafe { rgb_row_avx2(rgb, lut, out, blocks) };
    }
    let _ = avx2;
    // SAFETY: the pointer contract is the caller's.
    unsafe { rgb_row_body::<false>(rgb, lut, out, blocks) }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    //! Hand-written AVX2+FMA versions of the hot loops; the portable lane code beside them is the reference.
    //!
    //! The bilateral weight between two pixels is symmetric, so both 1-D filters compute it once per pixel
    //! pair `(i, i + d)` and use it for both ends: half the distance and `exp` work of a per-tap evaluation.
    use super::{CBRT_MAGIC, EXP2_POLY, Kern, LAB_EPS, LANES, TILE, V, VGeom, WHITE};
    use std::arch::x86_64::*;

    /// Largest radius with a specialised kernel; the radius is a const generic so the tap loops unroll.
    pub const MAX_RADIUS: usize = 8;

    macro_rules! with_radius {
        ($r:expr, $f:ident, $args:tt) => {
            match $r {
                1 => $f::<1> $args,
                2 => $f::<2> $args,
                3 => $f::<3> $args,
                4 => $f::<4> $args,
                5 => $f::<5> $args,
                6 => $f::<6> $args,
                7 => $f::<7> $args,
                8 => $f::<8> $args,
                _ => unreachable!("radius outside 1..=MAX_RADIUS"),
            }
        };
    }

    #[target_feature(enable = "avx2,fma")]
    fn splat(x: f32) -> __m256 {
        _mm256_set1_ps(x)
    }

    /// The polynomial coefficients of [`EXP2_POLY`] pre-multiplied by the spatial weight `s`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn scaled_poly(s: f32) -> [__m256; 5] {
        EXP2_POLY.map(|c| splat(c * s))
    }

    /// `s * 2^t` as in [`super::exp2_neg`], without the clamp: a lane whose `t` is out of range yields
    /// garbage, which [`weight`] discards through its cut-off mask.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn exp2_neg_scaled(t: __m256, c: &[__m256; 5]) -> __m256 {
        let ti = _mm256_cvttps_epi32(t);
        let f = _mm256_sub_ps(t, _mm256_cvtepi32_ps(ti));
        let mut p = c[4];
        p = _mm256_fmadd_ps(f, p, c[3]);
        p = _mm256_fmadd_ps(f, p, c[2]);
        p = _mm256_fmadd_ps(f, p, c[1]);
        p = _mm256_fmadd_ps(f, p, c[0]);
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(ti, _mm256_set1_epi32(127))));
        _mm256_mul_ps(scale, p)
    }

    /// `s * exp(-|a - b|^2 / 2 sigma_r^2)` (with `c = scaled_poly(s)`), zero at and beyond the cut-off.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn weight(a: [__m256; 3], b: [__m256; 3], c: &[__m256; 5], k: &Kern) -> __m256 {
        let d0 = _mm256_sub_ps(a[0], b[0]);
        let d1 = _mm256_sub_ps(a[1], b[1]);
        let d2 = _mm256_sub_ps(a[2], b[2]);
        let dd = _mm256_fmadd_ps(d2, d2, _mm256_fmadd_ps(d1, d1, _mm256_mul_ps(d0, d0)));
        let e = exp2_neg_scaled(_mm256_mul_ps(dd, splat(k.inv2)), c);
        _mm256_and_ps(e, _mm256_cmp_ps::<_CMP_LT_OQ>(dd, splat(k.max_d2)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn ld3(p: &[*const f32; 3], off: usize) -> [__m256; 3] {
        // SAFETY: forwarded from the caller.
        unsafe { [_mm256_loadu_ps(p[0].add(off)), _mm256_loadu_ps(p[1].add(off)), _mm256_loadu_ps(p[2].add(off))] }
    }

    /// Scratch floats per tap offset that [`h_row`] needs.
    pub fn h_scratch_len(r: usize) -> usize {
        r * (TILE + r + LANES)
    }

    /// Horizontal filter of one row: `blocks * LANES` outputs from planes padded by `r` samples per side.
    ///
    /// # Safety
    /// `src[ch]` points at the first (pad) sample of plane `ch`, readable up to `r + blocks * LANES + r + 7`;
    /// `dst[ch]` points at the first output sample, writable for `blocks * LANES`; `wbuf` has at least
    /// [`h_scratch_len`] floats. The CPU must support AVX2 and FMA.
    pub unsafe fn h_row(k: &Kern, src: [*const f32; 3], dst: [*mut f32; 3], blocks: usize, wbuf: &mut [f32]) {
        // SAFETY: forwarded; each instantiation is for the radius the match arm names.
        unsafe { with_radius!(k.r, h_row_r, (k, src, dst, blocks, wbuf)) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn h_row_r<const R: usize>(
        k: &Kern,
        src: [*const f32; 3],
        dst: [*mut f32; 3],
        blocks: usize,
        wbuf: &mut [f32],
    ) {
        let r = R;
        let ws = TILE + r + LANES;
        let wb = wbuf.as_mut_ptr();
        let tile_blocks = TILE / LANES;
        let mut t0 = 0;
        while t0 < blocks {
            let nb = (blocks - t0).min(tile_blocks);
            let i0 = r + t0 * LANES;
            // SAFETY: indices below stay within the padded planes and the scratch, per the contract.
            unsafe {
                for d in 1..=r {
                    let s = scaled_poly(k.sp[r + d]);
                    let wd = wb.add((d - 1) * ws);
                    for b in 0..(nb * LANES + d).div_ceil(LANES) {
                        let j = i0 - d + b * LANES;
                        let w = weight(ld3(&src, j), ld3(&src, j + d), &s, k);
                        _mm256_storeu_ps(wd.add(r - d + b * LANES), w);
                    }
                }
                for b in 0..nb {
                    let i = i0 + b * LANES;
                    let c = ld3(&src, i);
                    let mut acc = c;
                    let mut aw = splat(1.0);
                    for d in 1..=r {
                        let wd = wb.add((d - 1) * ws);
                        let wr = _mm256_loadu_ps(wd.add(r + b * LANES));
                        let wl = _mm256_loadu_ps(wd.add(r - d + b * LANES));
                        let qr = ld3(&src, i + d);
                        let ql = ld3(&src, i - d);
                        for ch in 0..3 {
                            acc[ch] = _mm256_fmadd_ps(wr, qr[ch], acc[ch]);
                            acc[ch] = _mm256_fmadd_ps(wl, ql[ch], acc[ch]);
                        }
                        aw = _mm256_add_ps(_mm256_add_ps(aw, wr), wl);
                    }
                    let iw = _mm256_div_ps(splat(1.0), aw);
                    for ch in 0..3 {
                        _mm256_storeu_ps(dst[ch].add((t0 + b) * LANES), _mm256_mul_ps(acc[ch], iw));
                    }
                }
            }
            t0 += tile_blocks;
        }
    }

    /// Scratch floats per tap offset pair that [`v_tile`] needs.
    pub fn v_scratch_len(r: usize) -> usize {
        (r * (r + 3) / 2) * TILE
    }

    /// Vertical filter of `nb * LANES` columns of output rows `y0..y1`, starting at the first sample of `src`
    /// and at `dst`. Rows outside the image repeat the nearest edge row.
    ///
    /// # Safety
    /// For every row `clamp(y, 0, h - 1)` with `y` in `y0 - r..y1 + r`, `src` must hold `nb * LANES` readable
    /// floats per plane; `dst` must be writable for the same extent in rows `y0..y1`; `nb * LANES <= TILE`,
    /// `y0 < y1 <= h`, and `ring` has at least [`v_scratch_len`] floats. The CPU must support AVX2 and FMA.
    pub unsafe fn v_tile(
        k: &Kern,
        src: *const f32,
        dst: *mut f32,
        geom: VGeom,
        nb: usize,
        rows: (usize, usize),
        ring: &mut [f32],
    ) {
        // SAFETY: forwarded; each instantiation is for the radius the match arm names.
        unsafe { with_radius!(k.r, v_tile_r, (k, src, dst, geom, nb, rows, ring)) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn v_tile_r<const R: usize>(
        k: &Kern,
        mid: *const f32,
        dst: *mut f32,
        g: VGeom,
        nb: usize,
        (y0, y1): (usize, usize),
        ring: &mut [f32],
    ) {
        let r = R;
        let rp = |y: isize| -> [*const f32; 3] {
            let y = y.clamp(0, g.h as isize - 1) as usize - g.y_first;
            // SAFETY: the row is inside `mid`.
            unsafe {
                [mid.add(y * g.src_row), mid.add(y * g.src_row + g.src_plane), mid.add(y * g.src_row + 2 * g.src_plane)]
            }
        };
        // Slot `(y + r) % (d + 1)` of tap offset `d`'s ring holds the weights of the row pair (y, y + d).
        let base = ring.as_mut_ptr();
        let slot = |d: usize, y: isize| -> *mut f32 {
            // SAFETY: the rings of offsets 1..=r use (d + 1) * TILE floats each, within `v_scratch_len`.
            unsafe { base.add(((d - 1) * (d + 2) / 2) * TILE + ((y + r as isize) as usize % (d + 1)) * TILE) }
        };
        let mut wr_p = [std::ptr::null::<f32>(); R];
        let mut wl_p = wr_p;
        let mut qr_p = [[std::ptr::null::<f32>(); 3]; R];
        let mut ql_p = qr_p;
        let compute = |d: usize, y: isize| {
            let (a, b) = (rp(y), rp(y + d as isize));
            let (s, out) = (scaled_poly(k.sp[r + d]), slot(d, y));
            for bl in 0..nb {
                // SAFETY: tile columns are inside the planes and the slot holds `TILE >= nb * LANES` floats.
                unsafe {
                    _mm256_storeu_ps(out.add(bl * LANES), weight(ld3(&a, bl * LANES), ld3(&b, bl * LANES), &s, k))
                };
            }
        };
        for d in 1..=r {
            for y in (y0 as isize - d as isize)..y0 as isize {
                compute(d, y);
            }
        }
        for y in y0 as isize..y1 as isize {
            for d in 1..=r {
                compute(d, y);
                wr_p[d - 1] = slot(d, y);
                wl_p[d - 1] = slot(d, y - d as isize);
                qr_p[d - 1] = rp(y + d as isize);
                ql_p[d - 1] = rp(y - d as isize);
            }
            let c = rp(y);
            let out = (y as usize - g.dst_y_first) * g.dst_row;
            for bl in 0..nb {
                let x = bl * LANES;
                // SAFETY: as above; the output row `y` is inside `dst`.
                unsafe {
                    let cc = ld3(&c, x);
                    let mut acc = cc;
                    let mut aw = splat(1.0);
                    for d in 0..r {
                        let wr = _mm256_loadu_ps(wr_p[d].add(x));
                        let wl = _mm256_loadu_ps(wl_p[d].add(x));
                        let qr = ld3(&qr_p[d], x);
                        let ql = ld3(&ql_p[d], x);
                        for ch in 0..3 {
                            acc[ch] = _mm256_fmadd_ps(wr, qr[ch], acc[ch]);
                            acc[ch] = _mm256_fmadd_ps(wl, ql[ch], acc[ch]);
                        }
                        aw = _mm256_add_ps(_mm256_add_ps(aw, wr), wl);
                    }
                    let iw = _mm256_div_ps(splat(1.0), aw);
                    for (ch, a) in acc.iter().enumerate() {
                        _mm256_storeu_ps(dst.add(out + ch * g.dst_plane + x), _mm256_mul_ps(*a, iw));
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn weight_matches_exp() {
            if !super::super::have_avx2() {
                return;
            }
            let k = Kern::new(&crate::smooth::BilateralParams::default());
            let s = 0.7f32;
            for i in 0..4000 {
                let dist = i as f32 * 0.0025 * k.max_d2.sqrt() * 1.1;
                // SAFETY: AVX2 and FMA were just detected.
                let w = unsafe {
                    let w = weight([splat(dist), splat(0.0), splat(0.0)], [splat(0.0); 3], &scaled_poly(s), &k);
                    let mut o = [0f32; 8];
                    _mm256_storeu_ps(o.as_mut_ptr(), w);
                    o[0]
                };
                let dd = dist * dist;
                let want = if dd < k.max_d2 { s * (-dd / (2.0 * 36.0)).exp() } else { 0.0 };
                assert!((w - want).abs() <= 1e-5 * want.max(1e-3), "dist {dist}: {w} vs {want}");
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn cbrt(x: __m256) -> __m256 {
        let g = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(x)), splat(1.0 / 3.0)));
        let mut y = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(CBRT_MAGIC)));
        for _ in 0..2 {
            let y3 = _mm256_mul_ps(_mm256_mul_ps(y, y), y);
            let num = _mm256_fmadd_ps(splat(2.0), x, y3);
            let den = _mm256_fmadd_ps(splat(2.0), y3, x);
            y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
        }
        y
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn lab_f(t: __m256) -> __m256 {
        let lin = _mm256_div_ps(_mm256_fmadd_ps(splat(24389.0 / 27.0), t, splat(16.0)), splat(116.0));
        let big = _mm256_cmp_ps::<_CMP_GT_OQ>(t, splat(LAB_EPS));
        _mm256_blendv_ps(lin, cbrt(t), big)
    }

    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn lab_math(lin: &[V; 3]) -> [V; 3] {
        // SAFETY: each `V` is eight readable floats.
        unsafe {
            let (r, g, b) =
                (_mm256_loadu_ps(lin[0].as_ptr()), _mm256_loadu_ps(lin[1].as_ptr()), _mm256_loadu_ps(lin[2].as_ptr()));
            let row = |a: f32, bb: f32, cc: f32, wh: f32| {
                let s = _mm256_fmadd_ps(splat(cc), b, _mm256_fmadd_ps(splat(bb), g, _mm256_mul_ps(splat(a), r)));
                lab_f(_mm256_mul_ps(s, splat(1.0 / wh)))
            };
            let fx = row(0.412_456_4, 0.357_576_1, 0.180_437_5, WHITE[0]);
            let fy = row(0.212_672_9, 0.715_152_2, 0.072_175_0, WHITE[1]);
            let fz = row(0.019_333_9, 0.119_192, 0.950_304_1, WHITE[2]);
            let l = _mm256_fmsub_ps(splat(116.0), fy, splat(16.0));
            let a = _mm256_mul_ps(splat(500.0), _mm256_sub_ps(fx, fy));
            let bl = _mm256_mul_ps(splat(200.0), _mm256_sub_ps(fy, fz));
            let mut o = [[0f32; LANES]; 3];
            _mm256_storeu_ps(o[0].as_mut_ptr(), l);
            _mm256_storeu_ps(o[1].as_mut_ptr(), a);
            _mm256_storeu_ps(o[2].as_mut_ptr(), bl);
            o
        }
    }
}

fn srgb_lut() -> [f32; 256] {
    std::array::from_fn(|v| srgb8_to_lin([v as u8; 3])[0])
}

/// sRGB image to Lab, with the sRGB decode from a 256-entry table and a vectorised Lab transform.
pub fn rgb_to_lab(img: &image::RgbImage) -> LabPlane {
    rgb_to_lab_with(img, have_avx2())
}

/// [`rgb_to_lab`] with the kernel choice made explicit, so tests can compare the SIMD and portable paths.
fn rgb_to_lab_with(img: &image::RgbImage, avx2: bool) -> LabPlane {
    let (w, h) = img.dimensions();
    let (wu, hu) = (w as usize, h as usize);
    let mut data = vec![[0f32; 3]; wu * hu];
    if wu == 0 || hu == 0 {
        return LabPlane::from_vec(w, h, data);
    }
    let lut = srgb_lut();
    let blocks = wu.div_ceil(LANES);
    let sw = blocks * LANES;
    data.par_chunks_mut(wu).zip(img.as_raw().par_chunks_exact(wu * 3)).for_each_init(
        || vec![0f32; 3 * sw],
        |planes, (orow, irow)| {
            let p = planes.as_mut_ptr();
            // SAFETY: `planes` holds three planes of `sw = blocks * LANES` floats.
            unsafe { rgb_row(avx2, irow, &lut, [p, p.add(sw), p.add(2 * sw)], blocks) };
            interleave(planes, sw, orow);
        },
    );
    LabPlane::from_vec(w, h, data)
}

/// Planar Lab (planes `stride` apart, first sample at index 0 of `planes`) to interleaved pixels.
fn interleave(planes: &[f32], stride: usize, out: &mut [[f32; 3]]) {
    let (p0, rest) = planes.split_at(stride);
    let (p1, p2) = rest.split_at(stride);
    for (x, o) in out.iter_mut().enumerate() {
        *o = [p0[x], p1[x], p2[x]];
    }
}

/// Lab plane to sRGB8.
pub fn lab_to_rgb(lab: &LabPlane) -> image::RgbImage {
    let raw: Vec<u8> = lab.data.par_iter().flat_map_iter(|&l| lab_to_srgb8(l)).collect();
    image::RgbImage::from_raw(lab.width, lab.height, raw).expect("buffer matches dims")
}

/// Row layout of the planar working buffers: each row holds the three Lab planes back to back, every plane
/// `radius` replicated samples, `blocks * LANES` samples (the width rounded up to whole vectors), `radius` more
/// replicated samples and `LANES` of slack so the kernels may read a whole vector past the pad.
struct Grid {
    r: usize,
    blocks: usize,
    /// Samples per plane in a row.
    plane: usize,
    /// Samples per row, all three planes.
    row: usize,
}

impl Grid {
    fn new(w: usize, r: usize) -> Self {
        let blocks = w.div_ceil(LANES);
        let plane = blocks * LANES + 2 * r + LANES;
        Grid { r, blocks, plane, row: 3 * plane }
    }
}

/// Raw pointer that tasks of one parallel region write through at disjoint places.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);
// SAFETY: every use writes regions no other task of the same region touches.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Read-only counterpart of [`Shared`].
#[derive(Clone, Copy)]
struct SharedConst<T>(*const T);
// SAFETY: the pointee is only read while the region runs.
unsafe impl<T> Send for SharedConst<T> {}
unsafe impl<T> Sync for SharedConst<T> {}

impl<T> SharedConst<T> {
    fn get(self) -> *const T {
        self.0
    }
}

/// Where the first pass reads its pixels from.
enum Source<'a> {
    Rgb(&'a image::RgbImage),
    Lab(&'a LabPlane),
}

/// Where [`v_tile`] reads and writes: `src` holds image rows `y_first..` of `src_row` floats, planes
/// `src_plane` apart, and covers every row the filter clamps to; `dst` starts at image row `dst_y_first`.
#[derive(Clone, Copy)]
struct VGeom {
    src_row: usize,
    src_plane: usize,
    y_first: usize,
    h: usize,
    dst_row: usize,
    dst_plane: usize,
    dst_y_first: usize,
}

/// Horizontal filter of `nb` vectors into a tile-local row.
///
/// # Safety
/// `src[ch]` points at the sample `radius` columns left of the first output and must be readable for
/// `nb * LANES + 2 * radius + LANES - 1` floats; `dst[ch]` is writable for `nb * LANES`; `nb * LANES <= TILE`;
/// `wbuf` has [`simd::h_scratch_len`] floats when `avx2`.
unsafe fn h_tile(avx2: bool, k: &Kern, src: [*const f32; 3], dst: [*mut f32; 3], nb: usize, wbuf: &mut [f32]) {
    debug_assert!(nb * LANES <= TILE);
    #[cfg(target_arch = "x86_64")]
    debug_assert!(!avx2 || wbuf.len() >= simd::h_scratch_len(k.r));
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: forwarded; `avx2` is only true after runtime detection.
        return unsafe { simd::h_row(k, src, dst, nb, wbuf) };
    }
    let _ = (avx2, &wbuf);
    // SAFETY: tap `d` reads `nb * LANES` samples from offset `d <= 2r`, within the contract's extent.
    unsafe {
        let taps: Vec<[*const f32; 3]> = (0..=2 * k.r).map(|d| [src[0].add(d), src[1].add(d), src[2].add(d)]).collect();
        filter_portable(k, &taps, dst, nb);
    }
}

/// Vertical filter of one tile (see [`simd::v_tile`]).
///
/// # Safety
/// As [`simd::v_tile`].
unsafe fn v_tile(
    avx2: bool,
    k: &Kern,
    (src, dst): (*const f32, *mut f32),
    g: VGeom,
    nb: usize,
    (y0, y1): (usize, usize),
    ring: &mut [f32],
) {
    debug_assert!(y0 < y1 && y1 <= g.h);
    debug_assert!(nb * LANES <= TILE);
    #[cfg(target_arch = "x86_64")]
    debug_assert!(!avx2 || ring.len() >= simd::v_scratch_len(k.r));
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: forwarded; `avx2` is only true after runtime detection.
        return unsafe { simd::v_tile(k, src, dst, g, nb, (y0, y1), ring) };
    }
    let _ = (avx2, &ring);
    for y in y0..y1 {
        // SAFETY: tap rows are clamped into the rows `src` holds, per the contract.
        unsafe {
            let taps: Vec<[*const f32; 3]> = (0..=2 * k.r)
                .map(|d| {
                    let rp = src.add(((y + d).saturating_sub(k.r).min(g.h - 1) - g.y_first) * g.src_row);
                    [rp, rp.add(g.src_plane), rp.add(2 * g.src_plane)]
                })
                .collect();
            let dp = dst.add((y - g.dst_y_first) * g.dst_row);
            filter_portable(k, &taps, [dp, dp.add(g.dst_plane), dp.add(2 * g.dst_plane)], nb);
        }
    }
}

/// Per-thread buffers of [`smooth_with`].
struct Scratch {
    /// Horizontally filtered rows of the current tile, three planes of `TILE` per row.
    mid: Vec<f32>,
    /// Source pixels of the first pass, three planes of `seg_plane`.
    seg: Vec<f32>,
    /// Result of the last pass for the current tile, before interleaving.
    out: Vec<f32>,
    h: Vec<f32>,
    v: Vec<f32>,
}

/// Bilateral filter: per pass a horizontal then a vertical 1-D filter, each in Lab with a Gaussian spatial
/// kernel and a Gaussian range kernel cut off at 3 sigma.
///
/// A pass runs as parallel tasks of [`V_ROWS`] rows by [`TILE`] columns: each filters the tile's rows plus
/// `radius` rows above and below horizontally into a cache-resident buffer, then vertically into the output,
/// so the intermediate image never goes through memory. Pass buffers are planar with padded rows ([`Grid`]).
/// The first pass converts sRGB to Lab on the fly and the last one writes the interleaved result.
fn smooth(src: Source, p: &BilateralParams) -> LabPlane {
    smooth_with(src, p, have_avx2())
}

/// [`smooth`] with the kernel choice made explicit, so tests can compare the SIMD and portable paths.
fn smooth_with(src: Source, p: &BilateralParams, avx2: bool) -> LabPlane {
    #[cfg(target_arch = "x86_64")]
    let avx2 = avx2 && (1..=simd::MAX_RADIUS).contains(&(p.radius as usize));
    let (w, h) = match src {
        Source::Rgb(i) => i.dimensions(),
        Source::Lab(l) => (l.width, l.height),
    };
    if w == 0 || h == 0 || p.passes == 0 {
        return match src {
            Source::Rgb(i) => rgb_to_lab(i),
            Source::Lab(l) => l.clone(),
        };
    }
    let (wu, hu) = (w as usize, h as usize);
    let k = Kern::new(p);
    let g = Grid::new(wu, k.r);
    let (r, plane, row, blocks) = (g.r, g.plane, g.row, g.blocks);
    let lut = srgb_lut();
    let sw = blocks * LANES;
    let seg_plane = TILE + 2 * r + 2 * LANES;
    let (h_scratch, v_scratch) = match avx2 {
        #[cfg(target_arch = "x86_64")]
        true => (simd::h_scratch_len(r), simd::v_scratch_len(r)),
        _ => (0, 0),
    };
    let tiles = sw.div_ceil(TILE);
    let chunks = hu.div_ceil(V_ROWS);
    // Pass `n` writes `bufs[n % 2]` (planar rows of `row` floats) and reads the other one; the last pass
    // interleaves into the front of its buffer, which becomes the result. A single pass needs one buffer.
    let passes = p.passes as usize;
    let mut bufs: Vec<Vec<[f32; 3]>> = (0..passes.min(2)).map(|_| vec![[0f32; 3]; hu * plane]).collect();

    for pass in 0..passes {
        let (first, last) = (pass == 0, pass + 1 == passes);
        let srcp = SharedConst(if first { std::ptr::null() } else { bufs[(pass + 1) % 2].as_ptr().cast::<f32>() });
        let dstp = Shared(bufs[pass % 2].as_mut_ptr().cast::<f32>());
        let outp = Shared(bufs[pass % 2].as_mut_ptr());
        (0..chunks * tiles).into_par_iter().for_each_init(
            || Scratch {
                mid: vec![0f32; (V_ROWS + 2 * r) * 3 * TILE],
                seg: vec![0f32; if first { 3 * seg_plane } else { 0 }],
                out: vec![0f32; if last { V_ROWS * 3 * TILE } else { 0 }],
                h: vec![0f32; h_scratch],
                v: vec![0f32; v_scratch],
            },
            |sc, idx| {
                let (y0, x0) = ((idx / tiles) * V_ROWS, (idx % tiles) * TILE);
                let y1 = (y0 + V_ROWS).min(hu);
                let nb = ((x0 + TILE).min(sw) - x0) / LANES;
                let (gs, ge) = (y0.saturating_sub(r), (y1 + r).min(hu));
                // SAFETY: every pointer below stays inside the buffers it came from (see the extents noted at
                // each helper); tasks write disjoint tiles of `dstp`/`outp` and only read `srcp`.
                unsafe {
                    for gy in gs..ge {
                        let s = if first {
                            fill_seg(&src, avx2, &lut, (gy, x0, nb), r, &mut sc.seg, seg_plane);
                            let sp = sc.seg.as_ptr();
                            [sp, sp.add(seg_plane), sp.add(2 * seg_plane)]
                        } else {
                            let sp = srcp.get().add(gy * row + x0);
                            [sp, sp.add(plane), sp.add(2 * plane)]
                        };
                        let m = sc.mid.as_mut_ptr().add((gy - gs) * 3 * TILE);
                        h_tile(avx2, &k, s, [m, m.add(TILE), m.add(2 * TILE)], nb, &mut sc.h);
                    }
                    let mut geom = VGeom {
                        src_row: 3 * TILE,
                        src_plane: TILE,
                        y_first: gs,
                        h: hu,
                        dst_row: row,
                        dst_plane: plane,
                        dst_y_first: 0,
                    };
                    let dst = if last {
                        geom.dst_row = 3 * TILE;
                        geom.dst_plane = TILE;
                        geom.dst_y_first = y0;
                        sc.out.as_mut_ptr()
                    } else {
                        dstp.get().add(r + x0)
                    };
                    v_tile(avx2, &k, (sc.mid.as_ptr(), dst), geom, nb, (y0, y1), &mut sc.v);
                    for y in y0..y1 {
                        if last {
                            let t = sc.out.as_ptr().add((y - y0) * 3 * TILE);
                            let o = outp.get().add(y * wu);
                            for x in 0..(nb * LANES).min(wu - x0) {
                                *o.add(x0 + x) = [*t.add(x), *t.add(TILE + x), *t.add(2 * TILE + x)];
                            }
                        } else {
                            for ch in 0..3 {
                                let pl = dstp.get().add(y * row + ch * plane);
                                if x0 == 0 {
                                    std::slice::from_raw_parts_mut(pl, r).fill(*pl.add(r));
                                }
                                if x0 + nb * LANES == sw {
                                    let v = *pl.add(r + wu - 1);
                                    std::slice::from_raw_parts_mut(pl.add(r + wu), plane - r - wu).fill(v);
                                }
                            }
                        }
                    }
                }
            },
        );
    }
    let mut out = std::mem::take(&mut bufs[(passes - 1) % 2]);
    out.truncate(wu * hu);
    LabPlane::from_vec(w, h, out)
}

/// Fills `seg` (three planes of `seg_plane`) with the first-pass input of image row `y`: Lab for columns
/// `x0 - r ..` (edge columns repeated), enough for a horizontal filter of `nb` vectors starting at column `x0`.
///
/// # Safety
/// `seg` must hold three planes of at least `nb * LANES + 2 * r + 2 * LANES` floats.
unsafe fn fill_seg(
    src: &Source,
    avx2: bool,
    lut: &[f32; 256],
    (y, x0, nb): (usize, usize, usize),
    r: usize,
    seg: &mut [f32],
    seg_plane: usize,
) {
    debug_assert!(seg.len() >= 3 * seg_plane);
    debug_assert!(nb * LANES + 2 * r + 2 * LANES <= seg_plane);
    let xs = x0.saturating_sub(r);
    let doff = xs + r - x0;
    let len = nb * LANES + 2 * r + LANES;
    let sp = seg.as_mut_ptr();
    let dst = [sp, sp.wrapping_add(seg_plane), sp.wrapping_add(2 * seg_plane)];
    match src {
        Source::Rgb(img) => {
            let w = img.width() as usize;
            // SAFETY: `blocks * LANES <= len - doff + 7`, inside the plane; the slice holds >= 1 pixel.
            unsafe {
                let dst = dst.map(|d| d.add(doff));
                rgb_row(avx2, &img.as_raw()[(y * w + xs) * 3..(y + 1) * w * 3], lut, dst, (len - doff).div_ceil(LANES));
            }
        }
        Source::Lab(lab) => {
            let w = lab.width as usize;
            for j in doff..len {
                let px = lab.data[y * w + (xs + j - doff).min(w - 1)];
                for ch in 0..3 {
                    seg[ch * seg_plane + j] = px[ch];
                }
            }
        }
    }
    for ch in 0..3 {
        let pl = &mut seg[ch * seg_plane..=ch * seg_plane + doff];
        let v = pl[doff];
        pl[..doff].fill(v);
    }
}

/// Bilateral filter in Lab with a Gaussian spatial kernel and a Gaussian range kernel cut off at 3 sigma,
/// applied as separate horizontal and vertical 1-D passes.
pub fn bilateral_lab(lab: &LabPlane, p: &BilateralParams) -> LabPlane {
    smooth(Source::Lab(lab), p)
}

/// [`rgb_to_lab`] followed by [`bilateral_lab`], with the colour conversion fused into the first pass so the
/// intermediate Lab plane is never materialised.
pub fn smooth_rgb(img: &image::RgbImage, p: &BilateralParams) -> LabPlane {
    smooth(Source::Rgb(img), p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::hash_u64;
    use crate::testutil::best;

    fn hash(x: u32, y: u32) -> u32 {
        (hash_u64(x as u64, y as u64, 0) >> 32) as u32
    }

    fn noisy(w: u32, h: u32) -> LabPlane {
        let mut lab = LabPlane::new(w, h, [0.0; 3]);
        for y in 0..h {
            for x in 0..w {
                let base = if x < w / 2 { 20.0 } else { 80.0 };
                let n = (hash(x, y) % 7) as f32 - 3.0;
                let m = (hash(y, x) % 5) as f32 - 2.0;
                lab.set(x, y, [base + n, m * 2.0 + (y % 9) as f32, -m]);
            }
        }
        lab
    }

    #[test]
    fn working_resize() {
        let big = image::RgbImage::new(4000, 3000);
        assert_eq!(to_working(&big, 2000).dimensions(), (2000, 1500));
        let small = image::RgbImage::new(800, 600);
        assert_eq!(to_working(&small, 2000).dimensions(), (800, 600));
    }

    #[test]
    fn bilateral_preserves_step_and_denoises() {
        let lab = noisy(64, 64);
        {
            let p = BilateralParams::default();
            let out = bilateral_lab(&lab, &p);
            for (lo, hi) in [(0u32, 30u32), (34, 64)] {
                let vals: Vec<f32> =
                    (0..64u32).flat_map(|y| (lo..hi).map(move |x| (x, y))).map(|(x, y)| out.get(x, y)[0]).collect();
                let m = vals.iter().sum::<f32>() / vals.len() as f32;
                let sd = (vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
                assert!(sd < 1.0, "std {sd}");
            }
            for y in 0..64 {
                assert!((out.get(30, y)[0] - 20.0).abs() < 3.0);
                assert!((out.get(33, y)[0] - 80.0).abs() < 3.0);
            }
        }
    }

    #[test]
    fn bilateral_constant() {
        let lab = LabPlane::new(20, 17, [50.0, 10.0, -20.0]);
        {
            let out = bilateral_lab(&lab, &BilateralParams::default());
            for (a, b) in out.data.iter().zip(&lab.data) {
                for i in 0..3 {
                    assert!((a[i] - b[i]).abs() < 1e-4);
                }
            }
        }
    }

    /// Plain per-pixel reference: edge-replicated horizontal then vertical window, true `exp`, taps summed in order.
    fn reference(lab: &LabPlane, p: &BilateralParams) -> LabPlane {
        let (w, h, r) = (lab.width as usize, lab.height as usize, p.radius as isize);
        let sp: Vec<f32> = (-r..=r).map(|d| (-((d * d) as f32) / (2.0 * p.sigma_s * p.sigma_s)).exp()).collect();
        let max_d2 = 9.0 * p.sigma_r * p.sigma_r;
        let one = |get: &dyn Fn(isize) -> [f32; 3], c: [f32; 3]| {
            let (mut s, mut sw) = ([0f32; 3], 0f32);
            for d in -r..=r {
                let q = get(d);
                let dd: f32 = (0..3).map(|k| (q[k] - c[k]).powi(2)).sum();
                if dd < max_d2 {
                    let wt = sp[(d + r) as usize] * (-dd / (2.0 * p.sigma_r * p.sigma_r)).exp();
                    for k in 0..3 {
                        s[k] += wt * q[k];
                    }
                    sw += wt;
                }
            }
            [s[0] / sw, s[1] / sw, s[2] / sw]
        };
        let mut cur = lab.clone();
        for _ in 0..p.passes {
            let mut mid = cur.clone();
            for y in 0..h {
                for x in 0..w {
                    let get = |d: isize| cur.data[y * w + (x as isize + d).clamp(0, w as isize - 1) as usize];
                    mid.data[y * w + x] = one(&get, cur.data[y * w + x]);
                }
            }
            let mut next = mid.clone();
            for y in 0..h {
                for x in 0..w {
                    let get = |d: isize| mid.data[(y as isize + d).clamp(0, h as isize - 1) as usize * w + x];
                    next.data[y * w + x] = one(&get, mid.data[y * w + x]);
                }
            }
            cur = next;
        }
        cur
    }

    fn max_diff(a: &LabPlane, b: &LabPlane) -> f32 {
        a.data.iter().zip(&b.data).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f32::max)
    }

    #[test]
    fn bilateral_matches_scalar_reference() {
        for (w, h) in [(64, 64), (37, 29), (8, 8), (5, 70), (300, 150)] {
            let lab = noisy(w, h);
            for p in [
                BilateralParams::default(),
                BilateralParams { radius: 2, sigma_s: 1.5, passes: 2, ..Default::default() },
                BilateralParams { radius: 0, passes: 1, ..Default::default() },
                BilateralParams { radius: 9, sigma_s: 4.0, passes: 1, ..Default::default() },
            ] {
                let d = max_diff(&bilateral_lab(&lab, &p), &reference(&lab, &p));
                assert!(d < 2e-3, "{w}x{h} {p:?}: max diff {d}");
            }
        }
    }

    #[test]
    fn simd_matches_portable() {
        if !have_avx2() {
            return;
        }
        for (w, h) in [(1u32, 1u32), (3, 70), (7, 9), (300, 150), (517, 131)] {
            let img = image::RgbImage::from_fn(w, h, |x, y| {
                let n = (hash_u64(x as u64, y as u64, 3) % 7) as u32;
                let base = if x < w / 2 { 60 } else { 190 };
                image::Rgb([(base + n) as u8, (base / 2 + (y % 9) * 2 + n) as u8, (base / 3 + n * 2) as u8])
            });
            let (a, b) = (rgb_to_lab_with(&img, false), rgb_to_lab_with(&img, true));
            assert!(max_diff(&a, &b) < 2e-3, "rgb_to_lab {w}x{h}");
            let lab = noisy(w, h);
            for radius in [1, 2, 5, 8] {
                for passes in [1, 3] {
                    let p = BilateralParams { radius, sigma_s: radius as f32 * 0.6, passes, ..Default::default() };
                    let (a, b) = (smooth_with(Source::Lab(&lab), &p, false), smooth_with(Source::Lab(&lab), &p, true));
                    assert!(max_diff(&a, &b) < 2e-3, "Lab {w}x{h} {p:?} {}", max_diff(&a, &b));
                    let (a, b) = (smooth_with(Source::Rgb(&img), &p, false), smooth_with(Source::Rgb(&img), &p, true));
                    assert!(max_diff(&a, &b) < 2e-3, "Rgb {w}x{h} {p:?} {}", max_diff(&a, &b));
                }
            }
        }
    }

    #[test]
    fn fused_rgb_matches_two_steps() {
        let img =
            image::RgbImage::from_fn(51, 33, |x, y| image::Rgb([(x * 5 + y) as u8, (x ^ y) as u8 * 3, (x * y) as u8]));
        let p = BilateralParams::default();
        let a = smooth_rgb(&img, &p);
        let b = bilateral_lab(&rgb_to_lab(&img), &p);
        assert!(max_diff(&a, &b) < 1e-3);
        let zero = BilateralParams { passes: 0, ..p };
        assert_eq!(smooth_rgb(&img, &zero), rgb_to_lab(&img));
    }

    #[test]
    fn bilateral_odd_shapes() {
        for (w, h) in [(1, 1), (1, 9), (9, 1), (3, 40), (40, 3)] {
            let lab = noisy(w, h);
            {
                let out = bilateral_lab(&lab, &BilateralParams::default());
                assert_eq!(out.data.len(), (w * h) as usize);
                assert!(out.data.iter().all(|v| v.iter().all(|c| c.is_finite())));
            }
        }
        assert_eq!(bilateral_lab(&LabPlane::new(0, 0, [0.0; 3]), &BilateralParams::default()).data.len(), 0);
    }

    #[test]
    fn resize_matches_image() {
        let img = image::RgbImage::from_fn(403, 302, |x, y| {
            image::Rgb([(x * 7 + y) as u8, (x ^ y) as u8, (x * y / 3) as u8])
        });
        for (nw, nh) in [(200, 150), (201, 151), (403, 151), (100, 302), (806, 604)] {
            let a = resize_triangle(&img, nw, nh);
            let b = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
            assert_eq!(a.dimensions(), b.dimensions());
            let maxd = a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap();
            assert!(maxd <= 1, "{nw}x{nh} maxd {maxd}");
        }
    }

    #[test]
    fn lab_lut_matches_ink() {
        let img = image::RgbImage::from_fn(256, 3, |x, y| {
            let v = x as u8;
            image::Rgb(match y {
                0 => [v, 0, 0],
                1 => [0, v, 0],
                _ => [0, 0, v],
            })
        });
        let lab = rgb_to_lab(&img);
        for (p, q) in img.pixels().zip(&lab.data) {
            let r = crate::ink::srgb8_to_lab(p.0);
            for k in 0..3 {
                assert!((q[k] - r[k]).abs() < 2e-3, "{:?} {q:?} vs {r:?}", p);
            }
        }
    }

    #[test]
    fn rgb_lab_roundtrip() {
        let mut img = image::RgbImage::new(4, 1);
        for (i, c) in [[0u8, 0, 0], [255, 255, 255], [200, 30, 90], [10, 120, 250]].iter().enumerate() {
            img.put_pixel(i as u32, 0, image::Rgb(*c));
        }
        let back = lab_to_rgb(&rgb_to_lab(&img));
        for (a, b) in img.pixels().zip(back.pixels()) {
            for i in 0..3 {
                assert!((a[i] as i32 - b[i] as i32).abs() <= 1);
            }
        }
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_bilateral() {
        let img = image::RgbImage::from_fn(2000, 1500, |x, y| {
            let n = hash(x, y) % 7;
            let b = 60 + 120 * ((x / 100 + y / 100) % 2);
            image::Rgb([(b + n) as u8, (b / 2 + n) as u8, (b / 3) as u8])
        });
        let lab = rgb_to_lab(&img);
        let (w, c) = best(|| rgb_to_lab(&img));
        eprintln!("rgb_to_lab: wall {w:.1} ms, cpu {c:.1} ms");
        for passes in [3, 5] {
            let p = BilateralParams { passes, ..Default::default() };
            let (w, c) = best(|| bilateral_lab(&lab, &p));
            let (fw, fc) = best(|| smooth_rgb(&img, &p));
            let (pw, pc) = best(|| smooth_with(Source::Rgb(&img), &p, false));
            eprintln!(
                "x{passes}: bilateral wall {w:.1} cpu {c:.1} ms; fused with rgb->Lab wall {fw:.1} cpu {fc:.1} ms; portable wall {pw:.1} cpu {pc:.1} ms"
            );
        }
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_resize_lab() {
        let img =
            image::RgbImage::from_fn(4032, 3024, |x, y| image::Rgb([(x ^ y) as u8, (x + y) as u8, (x * 3) as u8]));
        let (w, c) = best(|| resize_triangle(&img, 2000, 1500));
        eprintln!("resize 12MP->2000: wall {w:.1} ms, cpu {c:.1} ms");
    }
}
