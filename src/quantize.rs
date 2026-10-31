use crate::cc::label_runs;
use crate::codes::{PaletteLut, RenderedPalette, all_codes};
use crate::types::{IndexMap, LabPlane};
use rayon::prelude::*;

/// Lightness stretch: the 1st..99th percentile of L* maps onto `STRETCH_LO..STRETCH_HI`.
pub const STRETCH_LO: f32 = 5.0;
/// Upper end of the lightness stretch.
pub const STRETCH_HI: f32 = 97.0;
/// A narrower percentile span than this is widened (centred on the median) so low-contrast images are
/// stretched gently instead of having their noise blown up to full range.
pub const MIN_STRETCH_SPAN: f32 = 30.0;
/// Below this span the stretch is inactive: the image is flat and `boost_l` is the identity.
const FLAT_SPAN: f32 = 1.0;

/// Lightness stretch plus chroma gain, derived from the smoothed image.
#[derive(Clone, Copy, Debug)]
pub struct Boost {
    /// L* that maps to `STRETCH_LO`.
    pub l_lo: f32,
    /// L* that maps to `STRETCH_HI`.
    pub l_hi: f32,
    /// Chroma gain at and above `c_hi`.
    pub chroma: f32,
    /// Chroma gain at and below `c_lo`; below 1 it pulls near-neutrals toward paper.
    pub neutral_gain: f32,
    /// Chroma at and below which `neutral_gain` applies.
    pub c_lo: f32,
    /// Chroma at and above which `chroma` applies.
    pub c_hi: f32,
    /// Shadow-lift exponent applied after the linear stretch (1 = none, below 1 lifts darks).
    pub gamma: f32,
}

/// Dark-dominated images get a gamma lift so their median lands here (stretched L*); the palette has nothing
/// between full-overprint black (L 5-13) and L 37, so unlifted night scenes snap almost entirely to black.
const MEDIAN_TARGET: f32 = 32.0;
const MIN_GAMMA: f32 = 0.45;

/// Near-neutral chroma band: C* below `NEUTRAL_C_LO` gets `NEUTRAL_GAIN`, smoothstep up to full gain at `NEUTRAL_C_HI`.
pub const NEUTRAL_C_LO: f32 = 6.0;
/// Chroma at which the full chroma gain applies.
pub const NEUTRAL_C_HI: f32 = 40.0;
/// Chroma gain for near-neutrals.
pub const NEUTRAL_GAIN: f32 = 0.3;

/// Chroma ramp that lowers the black threshold for chromatic darks (see `lines::black_field`).
const RELIEF_C_LO: f32 = 5.0;
const RELIEF_C_HI: f32 = 18.0;

/// Smoothstep from 0 at `lo` to 1 at `hi`.
// f32::clamp asserts min <= max, which blocks auto-vectorisation of the per-pixel loops that call this.
#[allow(clippy::manual_clamp)]
#[inline]
pub fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    crate::noise::smooth01(((x - lo) / (hi - lo)).max(0.0).min(1.0))
}

/// Smoothstep weight of chroma `c` for the chromatic-dark black relief.
#[inline]
pub fn chroma_relief_ramp(c: f32) -> f32 {
    smoothstep(RELIEF_C_LO, RELIEF_C_HI, c)
}

impl Boost {
    /// Maps L 5..97 to itself, chroma 1.
    pub fn identity() -> Self {
        Boost {
            l_lo: STRETCH_LO,
            l_hi: STRETCH_HI,
            chroma: 1.0,
            neutral_gain: 1.0,
            c_lo: NEUTRAL_C_LO,
            c_hi: NEUTRAL_C_HI,
            gamma: 1.0,
        }
    }

    /// False for a flat image, where the stretch (and with it the shadow lift) is skipped everywhere.
    pub fn stretch_active(&self) -> bool {
        self.l_hi - self.l_lo >= FLAT_SPAN
    }
}

/// Rank of percentile `p` (0..1) among `n` sorted values.
pub fn percentile_rank(n: usize, p: f32) -> usize {
    ((p * (n.max(1) - 1) as f32).round() as usize).min(n.max(1) - 1)
}

/// Exact percentile `p` (0..1) of `vals`, which must not be empty.
pub fn percentile(vals: &[f32], p: f32) -> f32 {
    order_stats(vals, &[percentile_rank(vals.len(), p)])[0]
}

/// Percentiles of L* that drive the boost and the automatic black level.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightnessStats {
    /// 1st percentile.
    pub p01: f32,
    /// 8th percentile (automatic black level).
    pub p08: f32,
    /// Median.
    pub p50: f32,
    /// 99th percentile.
    pub p99: f32,
}

/// All four L* percentiles in one pass over the plane. An empty plane gives zeros.
pub fn lightness_stats(lab: &LabPlane) -> LightnessStats {
    if lab.data.is_empty() {
        return LightnessStats::default();
    }
    let ls: Vec<f32> = lab.data.par_iter().map(|p| p[0]).collect();
    let n = ls.len();
    let st = order_stats(
        &ls,
        &[percentile_rank(n, 0.01), percentile_rank(n, 0.08), percentile_rank(n, 0.5), percentile_rank(n, 0.99)],
    );
    LightnessStats { p01: st[0], p08: st[1], p50: st[2], p99: st[3] }
}

/// Boost from precomputed percentiles. A flat image (span under 1 L*) is left unstretched and unlifted.
pub fn boost_from_stats(st: &LightnessStats, chroma: f32) -> Boost {
    let mut b = Boost { l_lo: st.p01, l_hi: st.p99, chroma, neutral_gain: NEUTRAL_GAIN, ..Boost::identity() };
    if !b.stretch_active() {
        return b;
    }
    if b.l_hi - b.l_lo < MIN_STRETCH_SPAN {
        b.l_lo = st.p50 - MIN_STRETCH_SPAN / 2.0;
        b.l_hi = st.p50 + MIN_STRETCH_SPAN / 2.0;
    }
    let median = boost_l(st.p50, &b);
    if median < MEDIAN_TARGET && median > 1.0 {
        b.gamma = ((MEDIAN_TARGET / 100.0).ln() / (median / 100.0).ln()).clamp(MIN_GAMMA, 1.0);
    }
    b
}

/// Auto-levels from the 1st/99th percentile of L*, with the shadow lift for dark-dominated images.
pub fn compute_boost(lab: &LabPlane, chroma: f32) -> Boost {
    if lab.data.is_empty() {
        return Boost { chroma, neutral_gain: NEUTRAL_GAIN, ..Boost::identity() };
    }
    boost_from_stats(&lightness_stats(lab), chroma)
}

/// The boost's lightness stretch alone: l_lo..l_hi maps to 5..97.
#[allow(clippy::manual_clamp)]
#[inline]
pub fn boost_l(l: f32, b: &Boost) -> f32 {
    if !b.stretch_active() {
        return l;
    }
    let lin = (STRETCH_LO + (l - b.l_lo) * (STRETCH_HI - STRETCH_LO) / (b.l_hi - b.l_lo)).max(0.0).min(100.0);
    if b.gamma == 1.0 { lin } else { 100.0 * (lin / 100.0).powf(b.gamma) }
}

/// Inverse of [`boost_l`] (unclamped), for expressing a stretched-L threshold on unstretched L*.
pub fn unboost_l(l: f32, b: &Boost) -> f32 {
    if !b.stretch_active() {
        return l;
    }
    let lin = if b.gamma == 1.0 { l } else { 100.0 * (l / 100.0).powf(1.0 / b.gamma) };
    b.l_lo + (lin - STRETCH_LO) * (b.l_hi - b.l_lo) / (STRETCH_HI - STRETCH_LO)
}

/// A linear-stretch lightness carried through the shadow lift (for black thresholds defined on the stretch).
pub fn lifted(l: f32, b: &Boost) -> f32 {
    100.0 * (l / 100.0).powf(b.gamma)
}

/// Stretch lightness, scale chroma.
#[inline]
pub fn apply_boost(lab: [f32; 3], b: &Boost) -> [f32; 3] {
    let l = boost_l(lab[0], b);
    let gain = b.neutral_gain + (b.chroma - b.neutral_gain) * smoothstep(b.c_lo, b.c_hi, lab[1].hypot(lab[2]));
    [l, lab[1] * gain, lab[2] * gain]
}

/// [`apply_boost`] over a whole plane, in parallel.
pub fn boost_plane(lab: &LabPlane, b: &Boost) -> LabPlane {
    LabPlane::from_vec(lab.width, lab.height, lab.data.par_iter().map(|&p| apply_boost(p, b)).collect())
}

/// `lut.lookup` over a boosted plane, in parallel; the result is on the same grid.
pub fn snap_work(boosted: &LabPlane, lut: &PaletteLut) -> IndexMap {
    IndexMap::from_vec(boosted.width, boosted.height, boosted.data.par_iter().map(|&p| lut.lookup(p)).collect())
}

/// Exact order statistics (equal to `select_nth_unstable_by(total_cmp)` on a copy) for each rank in `ks`.
/// Values must be finite. A parallel histogram locates the bin, then one bin is resolved exactly.
pub fn order_stats(vals: &[f32], ks: &[usize]) -> Vec<f32> {
    const BINS: usize = 4096;
    assert!(ks.iter().all(|&k| k < vals.len()), "order_stats: rank out of range");
    if ks.is_empty() {
        return Vec::new();
    }
    let none = (f32::INFINITY, f32::NEG_INFINITY);
    let (lo, hi) = vals
        .par_iter()
        .fold(|| none, |(a, b), &v| (a.min(v), b.max(v)))
        .reduce(|| none, |x, y| (x.0.min(y.0), x.1.max(y.1)));
    let scale = if hi > lo { (BINS as f32 - 1.0) / (hi - lo) } else { 0.0 };
    let bin = |v: f32| (((v - lo) * scale) as usize).min(BINS - 1);
    let hist = vals
        .par_chunks(1 << 16)
        .map(|c| {
            let mut h = vec![0u32; BINS];
            for &v in c {
                h[bin(v)] += 1;
            }
            h
        })
        .reduce(
            || vec![0u32; BINS],
            |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            },
        );
    // (bin, rank within the bin) of each requested rank.
    let located: Vec<(usize, usize)> = ks
        .iter()
        .map(|&k| {
            let mut acc = 0usize;
            let mut b = 0;
            while acc + hist[b] as usize <= k {
                acc += hist[b] as usize;
                b += 1;
            }
            (b, k - acc)
        })
        .collect();
    let mut targets: Vec<usize> = located.iter().map(|&(b, _)| b).collect();
    targets.sort_unstable();
    targets.dedup();
    assert!(targets.len() < usize::from(u8::MAX), "order_stats: too many ranks");
    let mut slot = vec![u8::MAX; BINS];
    for (i, &b) in targets.iter().enumerate() {
        slot[b] = i as u8;
    }
    let none = || vec![Vec::new(); targets.len()];
    let mut inbin: Vec<Vec<f32>> = vals
        .par_chunks(1 << 16)
        .fold(none, |mut acc, c| {
            for &v in c {
                let s = slot[bin(v)];
                if s != u8::MAX {
                    acc[s as usize].push(v);
                }
            }
            acc
        })
        .reduce(none, |mut a, b| {
            for (x, mut y) in a.iter_mut().zip(b) {
                x.append(&mut y);
            }
            a
        });
    located
        .iter()
        .map(|&(b, r)| {
            let i = targets.binary_search(&b).expect("target bin");
            *inbin[i].select_nth_unstable_by(r, f32::total_cmp).1
        })
        .collect()
}

/// Majority filter over a square window of the given radius; a pixel changes only if another code beats its own.
pub fn mode_filter(map: &IndexMap, radius: u32, passes: u32) -> IndexMap {
    let mut cur = map.clone();
    if radius == 0 {
        return cur;
    }
    for _ in 0..passes {
        cur = mode_pass(&cur, radius as i64);
    }
    cur
}

fn mode_pass(src: &IndexMap, r: i64) -> IndexMap {
    let (w, h) = (src.width as i64, src.height as i64);
    let mut out = vec![0u8; src.data.len()];
    out.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
        let y = y as i64;
        let (y0, y1) = ((y - r).max(0), (y + r).min(h - 1));
        let mut hist = [0u32; 64];
        let mut total = 0u32;
        let column = |hist: &mut [u32; 64], total: &mut u32, x: i64, add: bool| {
            for yy in y0..=y1 {
                let v = src.data[(yy * w + x) as usize] as usize;
                if add {
                    hist[v] += 1;
                } else {
                    hist[v] -= 1;
                }
            }
            if add {
                *total += (y1 - y0 + 1) as u32;
            } else {
                *total -= (y1 - y0 + 1) as u32;
            }
        };
        for x in 0..=r.min(w - 1) {
            column(&mut hist, &mut total, x, true);
        }
        for x in 0..w {
            if x > 0 {
                if x + r < w {
                    column(&mut hist, &mut total, x + r, true);
                }
                if x > r {
                    column(&mut hist, &mut total, x - r - 1, false);
                }
            }
            let centre = src.data[(y * w + x) as usize];
            let cc = hist[centre as usize];
            row[x as usize] = if cc * 2 > total {
                centre
            } else {
                let mut best = centre;
                let mut best_c = cc;
                for (v, &c) in hist.iter().enumerate() {
                    if c > best_c {
                        best_c = c;
                        best = v as u8;
                    }
                }
                best
            };
        }
    });
    IndexMap::from_vec(src.width, src.height, out)
}

/// Merges 4-connected regions smaller than `min_area` into the neighbouring larger region that shares most of
/// their border, repeated up to `max_iters` times.
pub fn merge_small_regions(map: &mut IndexMap, min_area: u32, max_iters: u32) {
    let (w, h) = (map.width as usize, map.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    for _ in 0..max_iters {
        let rl = label_runs(&map.data, map.width, map.height, false, None);
        let area = rl.areas();
        if !area.iter().skip(1).any(|&a| a < min_area) {
            return;
        }
        let small = |l: u32| area[l as usize] < min_area;
        let mut slot = vec![u32::MAX; area.len()];
        let mut counts: Vec<[u32; 64]> = Vec::new();
        let mut add = |l: u32, v: u8, n: u32| {
            let s = &mut slot[l as usize];
            if *s == u32::MAX {
                *s = counts.len() as u32;
                counts.push([0; 64]);
            }
            counts[*s as usize][v as usize] += n;
        };
        for y in 0..h {
            let row = rl.row(y);
            for (k, r) in row.iter().enumerate() {
                if !small(r.label) {
                    continue;
                }
                let mut nb = |o: &crate::cc::Run, len: u32| {
                    if o.label != r.label && !small(o.label) {
                        add(r.label, o.val, len);
                    }
                };
                if k > 0 {
                    nb(&row[k - 1], 1);
                }
                if k + 1 < row.len() {
                    nb(&row[k + 1], 1);
                }
                for yy in [y.wrapping_sub(1), y + 1] {
                    if yy >= h {
                        continue;
                    }
                    for o in rl.row(yy) {
                        if o.x1 <= r.x0 {
                            continue;
                        }
                        if o.x0 >= r.x1 {
                            break;
                        }
                        nb(o, o.x1.min(r.x1) - o.x0.max(r.x0));
                    }
                }
            }
        }
        if counts.is_empty() {
            return;
        }
        let target: Vec<u8> = counts
            .iter()
            .map(|c| c.iter().enumerate().fold((0usize, 0u32), |b, (v, &n)| if n > b.1 { (v, n) } else { b }).0 as u8)
            .collect();
        map.data.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
            for r in rl.row(y) {
                let s = slot[r.label as usize];
                if s != u32::MAX {
                    line[r.x0 as usize..r.x1 as usize].fill(target[s as usize]);
                }
            }
        });
    }
}

/// Region-level re-pick. Each connected region of `map` gets the code of its mean boosted Lab (sampled from
/// the working grid, which may be a different size), then regions smaller than `min_area` take the code of the
/// adjacent larger region with the closest mean. Adjacent regions that end up with the same code merge implicitly.
pub fn repick_regions(map: &mut IndexMap, boosted_work: &LabPlane, lut: &PaletteLut, l_weight: f32, min_area: u32) {
    let (w, h) = (map.width as usize, map.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let rl = label_runs(&map.data, map.width, map.height, false, None);
    let n = rl.n_labels as usize;
    let area = rl.areas();
    let (ww, wh) = (boosted_work.width as usize, boosted_work.height as usize);
    let xs: Vec<usize> = (0..w).map(|x| (((x as f32 + 0.5) * ww as f32 / w as f32) as usize).min(ww - 1)).collect();
    // Summed per run in parallel, then per label in raster order, so the f64 totals are deterministic.
    let run_sums: Vec<[f64; 3]> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let wy = (((y as f32 + 0.5) * wh as f32 / h as f32) as usize).min(wh - 1);
            let wrow = &boosted_work.data[wy * ww..(wy + 1) * ww];
            let xs = &xs;
            rl.row(y).iter().map(move |r| {
                let mut s = [0f64; 3];
                for &x in &xs[r.x0 as usize..r.x1 as usize] {
                    let v = wrow[x];
                    s[0] += f64::from(v[0]);
                    s[1] += f64::from(v[1]);
                    s[2] += f64::from(v[2]);
                }
                s
            })
        })
        .collect();
    let mut sum = vec![[0f64; 3]; n];
    for (r, s) in rl.runs.iter().zip(&run_sums) {
        let t = &mut sum[r.label as usize];
        t[0] += s[0];
        t[1] += s[1];
        t[2] += s[2];
    }
    let mean: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let a = f64::from(area[i].max(1));
            [(sum[i][0] / a) as f32, (sum[i][1] / a) as f32, (sum[i][2] / a) as f32]
        })
        .collect();
    let mut code: Vec<u8> = mean.iter().map(|&m| lut.lookup(m)).collect();
    let orig = code.clone();

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for y in 0..h {
        let row = rl.row(y);
        for k in 1..row.len() {
            push_pair(&mut pairs, &area, min_area, row[k - 1].label, row[k].label);
        }
        if y + 1 < h {
            let below = rl.row(y + 1);
            let mut i = 0;
            for r in row {
                while i < below.len() && below[i].x1 <= r.x0 {
                    i += 1;
                }
                let mut k = i;
                while k < below.len() && below[k].x0 < r.x1 {
                    push_pair(&mut pairs, &area, min_area, r.label, below[k].label);
                    k += 1;
                }
            }
        }
    }
    // Sorted and deduped so ties between equally distant big neighbours resolve in label order.
    pairs.par_sort_unstable();
    pairs.dedup();
    let mut best: Vec<(f32, u8)> = vec![(f32::INFINITY, 0); n];
    for &(s, b) in &pairs {
        let d = crate::ink::delta_e(mean[s as usize], mean[b as usize], l_weight);
        if d < best[s as usize].0 {
            best[s as usize] = (d, orig[b as usize]);
        }
    }
    for (c, b) in code.iter_mut().zip(&best).skip(1) {
        if b.0.is_finite() {
            *c = b.1;
        }
    }
    map.data.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        for r in rl.row(y) {
            line[r.x0 as usize..r.x1 as usize].fill(code[r.label as usize]);
        }
    });
}

/// Records `(small, big)` for a pair of touching regions where exactly one side is under `min_area`.
fn push_pair(pairs: &mut Vec<(u32, u32)>, area: &[u32], min_area: u32, a: u32, b: u32) {
    if a != b {
        if area[a as usize] < min_area && area[b as usize] >= min_area {
            pairs.push((a, b));
        }
        if area[b as usize] < min_area && area[a as usize] >= min_area {
            pairs.push((b, a));
        }
    }
}

/// Samples fed to k-means (the plane is subsampled to about this many pixels).
const KMEANS_MAX_SAMPLES: usize = 160_000;
const KMEANS_MAX_ITERS: usize = 20;
const KMEANS_CONVERGENCE: f32 = 0.0025;

/// Fit k-means on boosted working Lab, snap each centroid to its nearest code; deduped, sorted.
pub fn kmeans_codes(
    lab_work: &LabPlane,
    boost: &Boost,
    k: usize,
    seed: u64,
    pal: &RenderedPalette,
    l_weight: f32,
) -> Vec<u8> {
    let stride = (lab_work.data.len() / KMEANS_MAX_SAMPLES).max(1);
    let samples: Vec<palette::Lab> = lab_work
        .data
        .iter()
        .step_by(stride)
        .map(|&p| {
            let b = apply_boost(p, boost);
            palette::Lab::new(b[0], b[1], b[2])
        })
        .collect();
    let km = kmeans_colors::get_kmeans_hamerly(k, KMEANS_MAX_ITERS, KMEANS_CONVERGENCE, false, &samples, seed);
    let all = all_codes();
    let mut codes: Vec<u8> = km.centroids.iter().map(|c| pal.nearest([c.l, c.a, c.b], l_weight, &all)).collect();
    codes.sort_unstable();
    codes.dedup();
    codes
}

#[cfg(test)]
mod reference {
    //! imageproc-based reference implementations, used as test oracles.
    use super::*;
    use image::{GrayImage, Luma};
    use imageproc::region_labelling::{Connectivity, connected_components};
    use std::collections::HashMap;

    pub fn merge_small_regions(map: &mut IndexMap, min_area: u32, max_iters: u32) {
        let (w, h) = (map.width as usize, map.height as usize);
        if w == 0 || h == 0 {
            return;
        }
        for _ in 0..max_iters {
            let img = GrayImage::from_raw(map.width, map.height, map.data.clone()).unwrap();
            let labels = connected_components(&img, Connectivity::Four, Luma([255u8]));
            let lab = labels.as_raw();
            let nlab = lab.iter().copied().max().unwrap_or(0) as usize + 1;
            let mut area = vec![0u32; nlab];
            for &l in lab {
                area[l as usize] += 1;
            }
            if !area.iter().skip(1).any(|&a| a < min_area) {
                return;
            }
            let mut counts: HashMap<u32, [u32; 64]> = HashMap::new();
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let l = lab[i];
                    if area[l as usize] >= min_area {
                        continue;
                    }
                    let mut visit = |j: usize| {
                        let nl = lab[j];
                        if nl != l && area[nl as usize] >= min_area {
                            counts.entry(l).or_insert([0; 64])[map.data[j] as usize] += 1;
                        }
                    };
                    if x > 0 {
                        visit(i - 1);
                    }
                    if x + 1 < w {
                        visit(i + 1);
                    }
                    if y > 0 {
                        visit(i - w);
                    }
                    if y + 1 < h {
                        visit(i + w);
                    }
                }
            }
            if counts.is_empty() {
                return;
            }
            let target: HashMap<u32, u8> = counts
                .iter()
                .map(|(&l, c)| {
                    let (v, _) =
                        c.iter().enumerate().fold((0usize, 0u32), |b, (v, &n)| if n > b.1 { (v, n) } else { b });
                    (l, v as u8)
                })
                .collect();
            for (i, &l) in lab.iter().enumerate() {
                if let Some(&v) = target.get(&l) {
                    map.data[i] = v;
                }
            }
        }
    }

    /// Region-level re-pick. Each connected region of `map` gets the code of its mean boosted Lab (sampled from
    /// the working grid), then regions smaller than `min_area` take the code of the adjacent larger region with the
    /// closest mean. Adjacent regions that end up with the same code merge implicitly.
    pub fn repick_regions(map: &mut IndexMap, boosted_work: &LabPlane, lut: &PaletteLut, l_weight: f32, min_area: u32) {
        let (w, h) = (map.width as usize, map.height as usize);
        if w == 0 || h == 0 {
            return;
        }
        let img = GrayImage::from_raw(map.width, map.height, map.data.clone()).unwrap();
        let labels = connected_components(&img, Connectivity::Four, Luma([255u8]));
        let lab = labels.as_raw();
        let n = lab.iter().copied().max().unwrap_or(0) as usize + 1;
        let (ww, wh) = (boosted_work.width as usize, boosted_work.height as usize);
        let xs: Vec<usize> = (0..w).map(|x| (((x as f32 + 0.5) * ww as f32 / w as f32) as usize).min(ww - 1)).collect();
        let mut area = vec![0u32; n];
        let mut sum = vec![[0f64; 3]; n];
        for y in 0..h {
            let wy = (((y as f32 + 0.5) * wh as f32 / h as f32) as usize).min(wh - 1);
            let wrow = &boosted_work.data[wy * ww..(wy + 1) * ww];
            for (x, &l) in lab[y * w..(y + 1) * w].iter().enumerate() {
                let v = wrow[xs[x]];
                let l = l as usize;
                area[l] += 1;
                for k in 0..3 {
                    sum[l][k] += v[k] as f64;
                }
            }
        }
        let mean: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let a = area[i].max(1) as f64;
                [(sum[i][0] / a) as f32, (sum[i][1] / a) as f32, (sum[i][2] / a) as f32]
            })
            .collect();
        let mut code: Vec<u8> = mean.iter().map(|&m| lut.lookup(m)).collect();

        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut note = |a: u32, b: u32| {
            if a != b {
                if area[a as usize] < min_area && area[b as usize] >= min_area {
                    pairs.push((a, b));
                }
                if area[b as usize] < min_area && area[a as usize] >= min_area {
                    pairs.push((b, a));
                }
            }
        };
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    note(lab[i], lab[i + 1]);
                }
                if y + 1 < h {
                    note(lab[i], lab[i + w]);
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let orig = code.clone();
        let mut best: Vec<(f32, u8)> = vec![(f32::INFINITY, 0); n];
        for &(s, b) in &pairs {
            let d = crate::ink::delta_e(mean[s as usize], mean[b as usize], l_weight);
            if d < best[s as usize].0 {
                best[s as usize] = (d, orig[b as usize]);
            }
        }
        for i in 1..n {
            if best[i].0.is_finite() {
                code[i] = best[i].1;
            }
        }
        for (o, &l) in map.data.iter_mut().zip(lab) {
            *o = code[l as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::test_palette;
    use crate::noise::hash_u64;
    use crate::types::Plane;

    fn pal() -> RenderedPalette {
        test_palette(crate::ink::PAPER_NEWSPRINT).1
    }

    fn flat_lab(l: f32) -> LabPlane {
        Plane::new(40, 40, [l, 0.0, 0.0])
    }

    #[test]
    fn boost_protects_near_neutrals() {
        let b = Boost {
            l_lo: 5.0,
            l_hi: 97.0,
            chroma: 1.3,
            neutral_gain: NEUTRAL_GAIN,
            c_lo: NEUTRAL_C_LO,
            c_hi: NEUTRAL_C_HI,
            gamma: 1.0,
        };
        let chroma = |lab: [f32; 3]| apply_boost(lab, &b)[1].hypot(apply_boost(lab, &b)[2]);
        assert!((chroma([50.0, 3.0, 4.0]) - 5.0 * NEUTRAL_GAIN).abs() < 1e-4);
        assert!((chroma([50.0, 48.0, 64.0]) - 80.0 * 1.3).abs() < 1e-3);
        let mid = chroma([50.0, 8.0, 6.0]) / 10.0;
        assert!(mid > NEUTRAL_GAIN && mid < 1.3);
        let id = apply_boost([50.0, 3.0, 4.0], &Boost::identity());
        assert_eq!((id[1], id[2]), (3.0, 4.0));
    }

    #[test]
    fn boost_identity_and_percentiles() {
        let id = Boost::identity();
        for l in [5.0f32, 20.0, 50.0, 97.0] {
            let o = apply_boost([l, 3.0, -4.0], &id);
            assert!((o[0] - l).abs() < 1e-4 && o[1] == 3.0 && o[2] == -4.0);
        }
        let data: Vec<[f32; 3]> = (0..10_000).map(|i| [20.0 + 40.0 * i as f32 / 9999.0, 0.0, 0.0]).collect();
        let b = compute_boost(&Plane::from_vec(100, 100, data), 1.3);
        assert!((b.l_lo - 20.4).abs() < 0.1 && (b.l_hi - 59.6).abs() < 0.1, "{b:?}");
        let o = apply_boost([20.0, 40.0, 0.0], &b);
        assert!((o[1] - 52.0).abs() < 1e-4);
    }

    #[test]
    fn flat_image_is_not_lifted() {
        // Gamma must come from the stretched median, so a flat image keeps gamma 1.
        for l in [20.0f32, 25.3, 26.0, 60.0] {
            let b = compute_boost(&flat_lab(l), 1.3);
            assert!(!b.stretch_active());
            assert_eq!(b.gamma, 1.0, "L {l}");
            assert_eq!(boost_l(l, &b), l);
            assert_eq!(lifted(25.0, &b), 25.0);
            assert_eq!(unboost_l(boost_l(l, &b), &b), l);
        }
        assert_eq!(compute_boost(&Plane::new(0, 0, [0.0; 3]), 1.3).gamma, 1.0);
    }

    #[test]
    fn low_contrast_stretch_gain_is_capped() {
        // Two tones 3 L* apart must not be stretched to black against white.
        let data: Vec<[f32; 3]> = (0..10_000).map(|i| [if i % 2 == 0 { 50.0 } else { 53.0 }, 0.0, 0.0]).collect();
        let b = compute_boost(&Plane::from_vec(100, 100, data), 1.0);
        assert!(b.stretch_active());
        let gain = (boost_l(53.0, &b) - boost_l(50.0, &b)) / 3.0;
        let max_gain = (STRETCH_HI - STRETCH_LO) / MIN_STRETCH_SPAN;
        assert!(gain <= max_gain + 1e-3, "gain {gain}");
        assert!(boost_l(50.0, &b) > 20.0 && boost_l(53.0, &b) < 80.0);
        let wide: Vec<[f32; 3]> = (0..10_000).map(|i| [20.0 + 40.0 * i as f32 / 9999.0, 0.0, 0.0]).collect();
        let w = compute_boost(&Plane::from_vec(100, 100, wide), 1.0);
        assert!((w.l_hi - w.l_lo - 39.2).abs() < 0.2, "wide images are untouched: {w:?}");
    }

    #[test]
    fn order_stats_match_select_nth() {
        let mut sets: Vec<Vec<f32>> = vec![
            (0..100_000u64).map(|i| (hash_u64(i, 1, 2) % 10007) as f32 * 0.013 - 7.0).collect(),
            (0..5000u64).map(|i| (hash_u64(i, 3, 4) % 5) as f32).collect(),
            vec![2.5; 777],
            vec![-1.0],
        ];
        sets.push((0..3000).map(|i| -(i as f32)).collect());
        for v in &sets {
            let n = v.len();
            let ks = [0, n / 100, n / 2, n * 99 / 100, n - 1];
            let got = order_stats(v, &ks);
            for (&k, g) in ks.iter().zip(got) {
                let mut c = v.clone();
                assert_eq!(c.select_nth_unstable_by(k, f32::total_cmp).1.to_bits(), g.to_bits(), "n {n} k {k}");
            }
        }
        assert!(order_stats(&[], &[]).is_empty());
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5), 2.0);
    }

    #[test]
    fn lightness_stats_agree_with_boost() {
        let data: Vec<[f32; 3]> =
            (0..10_000).map(|i| [10.0 + 80.0 * ((i * 37) % 10_000) as f32 / 9999.0, 0.0, 0.0]).collect();
        let lab = Plane::from_vec(100, 100, data);
        let st = lightness_stats(&lab);
        assert!(st.p01 < st.p08 && st.p08 < st.p50 && st.p50 < st.p99);
        let b = compute_boost(&lab, 1.3);
        assert_eq!((b.l_lo, b.l_hi), (st.p01, st.p99));
    }

    #[test]
    fn snap_flat() {
        let pal = pal();
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        for k in [0usize, 21, 63] {
            let plane = Plane::new(10, 8, pal.lab[k]);
            let w = snap_work(&boost_plane(&plane, &Boost::identity()), &lut);
            assert_eq!((w.width, w.height), (10, 8));
            assert!(w.data.iter().all(|&v| v as usize == k));
        }
    }

    #[test]
    fn mode_filter_props() {
        let mut m = Plane::new(40, 40, 9u8);
        m.set(20, 20, 5);
        assert!(mode_filter(&m, 2, 1).data.iter().all(|&v| v == 9));
        let mut b = Plane::new(40, 40, 3u8);
        for y in 0..40 {
            for x in 20..40 {
                b.set(x, y, 7);
            }
        }
        assert_eq!(mode_filter(&b, 2, 2), b);
        let row = Plane::from_vec(7, 1, vec![1u8, 1, 2, 1, 1, 2, 2]);
        assert_eq!(mode_filter(&row, 1, 1).data, vec![1, 1, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn merge_islands() {
        let mut m = Plane::new(100, 100, 9u8);
        for y in 10..13 {
            for x in 10..13 {
                m.set(x, y, 5);
            }
        }
        for y in 50..60 {
            for x in 50..60 {
                m.set(x, y, 5);
            }
        }
        merge_small_regions(&mut m, 20, 3);
        assert_eq!(m.get(11, 11), 9);
        assert_eq!(m.get(55, 55), 5);
    }

    #[test]
    fn repick_makes_noisy_region_one_code() {
        let pal = pal();
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        let (a, b) = (3u8, 15u8);
        let target = lut.lookup(pal.lab[a as usize]);
        let work = Plane::new(40, 40, pal.lab[a as usize]);
        let mut m = Plane::new(80, 80, a);
        for y in 0..80usize {
            for x in 0..80usize {
                if (x * 7 + y * 13) % 11 < 3 && x % 4 < 2 {
                    m.data[y * 80 + x] = b;
                }
            }
        }
        let noisy_a = m.data.iter().map(|&v| usize::from(v == b)).sum::<usize>();
        assert!(noisy_a > 100);
        repick_regions(&mut m, &work, &lut, 1.5, 200);
        assert!(m.data.iter().all(|&v| v == target), "region still holds several codes");
    }

    #[test]
    fn region_stages_survive_all_distinct_neighbours() {
        // A map where every pixel is its own region must not panic.
        let pal = pal();
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        let work = Plane::new(2, 2, pal.lab[3]);
        let cases = [
            Plane::from_vec(1, 1, vec![3u8]),
            Plane::from_vec(2, 1, vec![3u8, 15]),
            Plane::from_vec(1, 2, vec![3u8, 15]),
            Plane::from_vec(64, 64, (0..64 * 64).map(|i| if (i % 64 + i / 64) % 2 == 0 { 3u8 } else { 15 }).collect()),
        ];
        for m in cases {
            let mut a = m.clone();
            merge_small_regions(&mut a, 2, 3);
            assert!(a.data.iter().all(|&v| v == 3 || v == 15));
            let mut b = m.clone();
            repick_regions(&mut b, &work, &lut, 1.5, 4);
            assert_eq!((b.width, b.height), (m.width, m.height));
        }
        let mut one = Plane::from_vec(1, 1, vec![3u8]);
        merge_small_regions(&mut one, 5, 3);
        assert_eq!(one.data, vec![3]);
    }

    #[test]
    fn region_stages_match_reference() {
        let pal = pal();
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        for seed in 0..12 {
            let (w, h) = (90u32, 70u32);
            let d: Vec<u8> = (0..w * h)
                .map(|i| {
                    (hash_u64(u64::from(i % w / 4), u64::from(i / w / 3), seed) % 5) as u8 * 7
                        + u8::from(hash_u64(u64::from(i), 9, seed).is_multiple_of(9))
                })
                .collect();
            let mut a = Plane::from_vec(w, h, d);
            let mut b = a.clone();
            reference::merge_small_regions(&mut a, 12, 3);
            merge_small_regions(&mut b, 12, 3);
            assert_eq!(a, b, "merge seed {seed}");
            let work = Plane::from_vec(45, 35, (0..45 * 35).map(|i| pal.lab[(i * 7 + seed as usize) % 64]).collect());
            reference::repick_regions(&mut a, &work, &lut, 1.5, 40);
            repick_regions(&mut b, &work, &lut, 1.5, 40);
            assert_eq!(a, b, "repick seed {seed}");
        }
    }

    #[test]
    fn boost_l_round_trips_with_lift() {
        let mut b = Boost::identity();
        b.l_lo = 2.0;
        b.l_hi = 80.0;
        b.gamma = 0.6;
        for l in [10.0f32, 30.0, 60.0] {
            assert!((unboost_l(boost_l(l, &b), &b) - l).abs() < 1e-3);
        }
        assert!(boost_l(10.0, &b) > 5.0 + 8.0 * 92.0 / 78.0);
    }

    #[test]
    fn dark_images_get_a_lift() {
        let data: Vec<[f32; 3]> =
            (0..10_000).map(|i| [if i < 9800 { 1.0 + 8.0 * i as f32 / 9800.0 } else { 90.0 }, 0.0, 0.0]).collect();
        let b = compute_boost(&Plane::from_vec(100, 100, data), 1.3);
        assert!(b.gamma < 1.0);
        let bright: Vec<[f32; 3]> = (0..10_000).map(|i| [30.0 + 60.0 * i as f32 / 9999.0, 0.0, 0.0]).collect();
        assert_eq!(compute_boost(&Plane::from_vec(100, 100, bright), 1.3).gamma, 1.0);
    }

    #[test]
    fn kmeans_three_flat() {
        let pal = pal();
        let codes = [3usize, 20, 50];
        let mut data = Vec::new();
        for i in 0..3000 {
            data.push(pal.lab[codes[i % 3]]);
        }
        let plane = Plane::from_vec(60, 50, data);
        let got = kmeans_codes(&plane, &Boost::identity(), 3, 1, &pal, 1.5);
        assert_eq!(got, vec![3, 20, 50]);
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_regions() {
        use crate::testutil::time;
        let pal = pal();
        let lut = PaletteLut::new(&pal, 1.5, &all_codes());
        let (w, h) = (2016u32, 1512u32);
        let mut data = Vec::new();
        for i in 0..w * h {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            data.push([
                50.0 + 40.0 * (x * 0.01).sin() * (y * 0.013).cos(),
                30.0 * (x * 0.02).sin(),
                30.0 * (y * 0.02).cos(),
            ]);
        }
        let plane = Plane::from_vec(w, h, data);
        let st = time("lightness_stats", || lightness_stats(&plane));
        let b = boost_from_stats(&st, 1.3);
        let bw = time("boost_plane", || boost_plane(&plane, &b));
        let m = time("snap_work", || snap_work(&bw, &lut));
        let mut m = time("mode", || mode_filter(&m, 1, 1));
        time("merge", || merge_small_regions(&mut m, 60, 3));
        time("repick", || repick_regions(&mut m, &bw, &lut, 1.5, 120));
        time("kmeans", || kmeans_codes(&plane, &b, 12, 1, &pal, 1.5));
    }
}
