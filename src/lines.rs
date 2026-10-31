use crate::cc::label_runs;
use crate::morph::{disk_close, disk_dilate, disk_open};
use crate::quantize::{Boost, boost_l, chroma_relief_ramp};
use crate::types::{FloatPlane, LabPlane};
use rayon::prelude::*;

/// Output short edge (px) at which the line and black parameters are tuned; larger outputs scale up from it.
pub const REFERENCE_SHORT_EDGE: f32 = 3000.0;
/// Smallest line fragment kept, in output px at the reference size (8-connected area).
pub const LINE_MIN_AREA: f32 = 120.0;
/// Line thickening radius as a fraction of the output short edge.
pub const LINE_DILATE_FRAC: f32 = 0.001;
/// Smallest black blob and hole kept, as a fraction of the output area.
pub const BLACK_AREA_FRAC: f32 = 0.0005;
/// Opening radius that removes specks from the black mask, in output px at the reference size.
pub const BLACK_OPEN_R: f32 = 2.0;
/// Closing radius that joins nearby black blobs, in output px at the reference size.
pub const BLACK_CLOSE_R: f32 = 3.0;
/// Chromatic darks (a dark green background) get this much lower black threshold than neutral ones (a black bag).
pub const CHROMA_RELIEF: f32 = 14.0;

/// Resolution scale of the output relative to [`REFERENCE_SHORT_EDGE`], never below 1.
pub fn output_scale(out_w: u32, out_h: u32) -> f32 {
    (out_w.min(out_h) as f32 / REFERENCE_SHORT_EDGE).max(1.0)
}

/// Geometry that turns output-pixel parameters into work-grid ones.
#[derive(Clone, Copy, Debug)]
pub struct MaskScale {
    /// Output width in px.
    pub out_w: u32,
    /// Output height in px.
    pub out_h: u32,
    /// [`output_scale`] of the output.
    pub s: f32,
    /// Work width divided by output width.
    pub f: f32,
}

impl MaskScale {
    /// `work_w` is the width of the grid the masks are computed on.
    pub fn new(out_w: u32, out_h: u32, work_w: u32) -> Self {
        MaskScale { out_w, out_h, s: output_scale(out_w, out_h), f: work_w as f32 / out_w as f32 }
    }
}

/// Difference-of-Gaussians edge detector parameters.
pub struct XdogParams {
    /// Inner Gaussian sigma in work px.
    pub sigma: f32,
    /// Outer sigma is `sigma * k`.
    pub k: f32,
    /// Sharpening weight of the difference.
    pub p: f32,
}

impl Default for XdogParams {
    fn default() -> Self {
        XdogParams { sigma: 1.2, k: 1.6, p: 20.0 }
    }
}

/// Separable Gaussian with clamped borders, kernel radius ceil(3 sigma).
fn gaussian(src: &FloatPlane, sigma: f32) -> FloatPlane {
    let (w, h) = (src.width as usize, src.height as usize);
    if w == 0 || h == 0 {
        return src.clone();
    }
    let r = (3.0 * sigma).ceil().max(1.0) as usize;
    let mut k: Vec<f32> = (0..=2 * r)
        .map(|i| {
            let d = i as f32 - r as f32;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }

    let mut tmp = vec![0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each_init(
        || Vec::with_capacity(w + 2 * r),
        |pad, (y, row)| {
            let line = &src.data[y * w..(y + 1) * w];
            pad.clear();
            pad.resize(r, line[0]);
            pad.extend_from_slice(line);
            pad.resize(w + 2 * r, line[w - 1]);
            for (i, &kv) in k.iter().enumerate() {
                for (o, &v) in row.iter_mut().zip(&pad[i..i + w]) {
                    *o += kv * v;
                }
            }
        },
    );
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (i, &kv) in k.iter().enumerate() {
            let sy = (y + i).saturating_sub(r).min(h - 1);
            for (o, &v) in row.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                *o += kv * v;
            }
        }
    });
    FloatPlane::from_vec(src.width, src.height, out)
}

/// Sharpened difference-of-Gaussians edge field on the working grid.
pub fn xdog_field(lab: &LabPlane, p: &XdogParams) -> FloatPlane {
    let u = lightness_plane(lab);
    let g1 = gaussian(&u, p.sigma);
    let g2 = gaussian(&u, p.sigma * p.k);
    let data = g1.data.par_iter().zip(g2.data.par_iter()).map(|(a, b)| (1.0 + p.p) * a - p.p * b).collect();
    FloatPlane::from_vec(u.width, u.height, data)
}

fn lightness_plane(lab: &LabPlane) -> FloatPlane {
    let data = lab.data.par_iter().map(|l| l[0] / 100.0).collect();
    FloatPlane::from_vec(lab.width, lab.height, data)
}

/// Stretched lightness (L*/100) with chromatic darks lifted by [`CHROMA_RELIEF`], the field the black mask
/// thresholds.
pub fn black_field(lab: &LabPlane, boost: &Boost) -> FloatPlane {
    let data = lab
        .data
        .par_iter()
        .map(|v| (boost_l(v[0], boost) + CHROMA_RELIEF * chroma_relief_ramp(v[1].hypot(v[2]))) / 100.0)
        .collect();
    FloatPlane::from_vec(lab.width, lab.height, data)
}

/// Clears 8-connected foreground components smaller than `min_area` in a raw 0/255 buffer.
pub fn remove_small_buf(mask: &mut [u8], w: u32, h: u32, min_area: u32) {
    if w == 0 || h == 0 {
        return;
    }
    let rl = label_runs(mask, w, h, true, Some(0));
    let area = rl.areas();
    mask.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
        for r in rl.row(y) {
            if area[r.label as usize] < min_area {
                row[r.x0 as usize..r.x1 as usize].fill(0);
            }
        }
    });
}

/// Fills 4-connected background components smaller than `max_area` that do not touch the border, in a raw
/// 0/255 buffer.
pub fn fill_holes_buf(mask: &mut [u8], w: u32, h: u32, max_area: u32) {
    if w == 0 || h == 0 {
        return;
    }
    let rl = label_runs(mask, w, h, false, Some(255));
    let area = rl.areas();
    let border = rl.touches_border();
    mask.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
        for r in rl.row(y) {
            let l = r.label as usize;
            if !border[l] && area[l] < max_area {
                row[r.x0 as usize..r.x1 as usize].fill(255);
            }
        }
    });
}

/// Parameters of the line mask, in work-grid pixels.
#[derive(Clone, Copy, Debug)]
pub struct MaskParams {
    /// Line fragments smaller than this (8-connected) are dropped.
    pub min_area: u32,
    /// Closing disk radius; 0 = none.
    pub close_r: f32,
    /// Dilation disk radius; 0 = none.
    pub dilate_r: f32,
}

impl MaskParams {
    /// Work-grid parameters from the output-pixel ones. `area_mul` scales the minimum fragment area and
    /// `dilate_mul` the line thickness.
    pub fn scaled(g: &MaskScale, area_mul: f32, dilate_mul: f32) -> Self {
        let min_area = (LINE_MIN_AREA * area_mul * g.s * g.s * g.f * g.f) as u32;
        // The close is an LInf disk of radius round(s) in output px; it only survives if it rounds to a work px.
        let sf = (g.s * g.f).round();
        let close_r = if sf >= 1.0 { 1.5 * sf } else { 0.0 };
        let short = g.out_w.min(g.out_h) as f32;
        let k = (LINE_DILATE_FRAC * short * dilate_mul).round() * g.f;
        let dilate_r = if k > 0.0 { k.max(1.0) } else { 0.0 };
        MaskParams { min_area, close_r, dilate_r }
    }
}

/// Parameters of the black mask, in work-grid pixels.
#[derive(Clone, Copy, Debug)]
pub struct BlackParams {
    /// Opening disk radius.
    pub open_r: f32,
    /// Closing disk radius.
    pub close_r: f32,
    /// Blobs and holes smaller than this are removed or filled.
    pub area: u32,
}

impl BlackParams {
    /// Work-grid parameters from the output-pixel ones.
    pub fn scaled(g: &MaskScale) -> Self {
        BlackParams {
            open_r: (BLACK_OPEN_R * g.s * g.f).max(1.0),
            close_r: BLACK_CLOSE_R * g.s * g.f,
            area: (g.out_w as f32 * g.out_h as f32 * BLACK_AREA_FRAC * g.f * g.f) as u32,
        }
    }
}

fn clean_line_buf(mut m: Vec<u8>, w: u32, h: u32, p: &MaskParams) -> Vec<u8> {
    remove_small_buf(&mut m, w, h, p.min_area);
    if p.close_r > 0.0 {
        m = disk_close(&m, w, h, p.close_r);
    }
    if p.dilate_r > 0.0 {
        m = disk_dilate(&m, w, h, p.dilate_r);
    }
    m
}

fn clean_black_buf(mut m: Vec<u8>, w: u32, h: u32, p: &BlackParams) -> Vec<u8> {
    if p.open_r > 0.0 {
        m = disk_open(&m, w, h, p.open_r);
    }
    if p.close_r > 0.0 {
        m = disk_close(&m, w, h, p.close_r);
    }
    remove_small_buf(&mut m, w, h, p.area);
    fill_holes_buf(&mut m, w, h, p.area);
    m
}

fn below_mask(field: &FloatPlane, below: f32) -> Vec<u8> {
    field.data.par_iter().map(|&v| if v < below { 255 } else { 0 }).collect()
}

/// Line mask (0/255) on the work grid: pixels of `xdog` under `below`, small fragments removed, then closed and
/// thickened.
pub fn line_mask_work(xdog: &FloatPlane, below: f32, p: &MaskParams) -> Vec<u8> {
    clean_line_buf(below_mask(xdog, below), xdog.width, xdog.height, p)
}

/// Spotted-black mask (0/255) on the work grid: pixels of `stretched` under `below`, opened, closed, with small
/// blobs removed and small holes filled.
pub fn black_mask_work(stretched: &FloatPlane, below: f32, p: &BlackParams) -> Vec<u8> {
    clean_black_buf(below_mask(stretched, below), stretched.width, stretched.height, p)
}

#[cfg(test)]
mod reference {
    //! imageproc-based reference implementations, used as test oracles.
    use crate::types::{FloatPlane, Mask};
    use image::{ImageBuffer, Luma};
    use imageproc::region_labelling::{Connectivity, connected_components};
    use rayon::prelude::*;

    pub fn gaussian(src: &FloatPlane, sigma: f32) -> FloatPlane {
        let (w, h) = (src.width as usize, src.height as usize);
        let r = (3.0 * sigma).ceil().max(1.0) as usize;
        let mut k: Vec<f32> = (0..=2 * r)
            .map(|i| {
                let d = i as f32 - r as f32;
                (-d * d / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let sum: f32 = k.iter().sum();
        for v in &mut k {
            *v /= sum;
        }
        let mut tmp = vec![0f32; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let line = &src.data[y * w..(y + 1) * w];
            for (x, o) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let sx = (x as i64 + i as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                    acc += kv * line[sx];
                }
                *o = acc;
            }
        });
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (i, kv) in k.iter().enumerate() {
                let sy = (y as i64 + i as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                let line = &tmp[sy * w..(sy + 1) * w];
                for (o, v) in row.iter_mut().zip(line) {
                    *o += kv * v;
                }
            }
        });
        FloatPlane::from_vec(src.width, src.height, out)
    }

    fn component_stats(labels: &ImageBuffer<Luma<u32>, Vec<u32>>) -> (Vec<u32>, Vec<bool>) {
        let n = labels.as_raw().iter().copied().max().unwrap_or(0) as usize + 1;
        let mut area = vec![0u32; n];
        let mut border = vec![false; n];
        let (w, h) = labels.dimensions();
        for (i, &l) in labels.as_raw().iter().enumerate() {
            area[l as usize] += 1;
            let (x, y) = (i as u32 % w, i as u32 / w);
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                border[l as usize] = true;
            }
        }
        (area, border)
    }

    pub fn remove_small_components(mask: &mut Mask, min_area: u32, eight: bool) {
        let conn = if eight { Connectivity::Eight } else { Connectivity::Four };
        let labels = connected_components(&*mask, conn, Luma([0u8]));
        let (area, _) = component_stats(&labels);
        for (m, &l) in mask.as_mut().iter_mut().zip(labels.as_raw()) {
            if l != 0 && area[l as usize] < min_area {
                *m = 0;
            }
        }
    }

    pub fn fill_small_holes(mask: &mut Mask, max_area: u32) {
        let labels = connected_components(&*mask, Connectivity::Four, Luma([255u8]));
        let (area, border) = component_stats(&labels);
        for (m, &l) in mask.as_mut().iter_mut().zip(labels.as_raw()) {
            if l != 0 && !border[l as usize] && area[l as usize] < max_area {
                *m = 255;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::hash_u64;
    use crate::types::{Mask, Plane};
    use image::{GrayImage, Luma};

    fn threshold(field: &FloatPlane, below: f32) -> Mask {
        GrayImage::from_raw(field.width, field.height, below_mask(field, below)).unwrap()
    }

    fn remove_small_components(mask: &mut Mask, min_area: u32) {
        let (w, h) = mask.dimensions();
        remove_small_buf(mask.as_mut(), w, h, min_area);
    }

    fn fill_small_holes(mask: &mut Mask, max_area: u32) {
        let (w, h) = mask.dimensions();
        fill_holes_buf(mask.as_mut(), w, h, max_area);
    }

    fn clean_lines(mask: Mask, area_mul: f32, dilate_mul: f32) -> Mask {
        let (w, h) = mask.dimensions();
        let p = MaskParams::scaled(&MaskScale::new(w, h, w), area_mul, dilate_mul);
        GrayImage::from_raw(w, h, clean_line_buf(mask.into_raw(), w, h, &p)).unwrap()
    }

    fn spotted_blacks(mask: Mask) -> Mask {
        let (w, h) = mask.dimensions();
        let p = BlackParams::scaled(&MaskScale::new(w, h, w));
        GrayImage::from_raw(w, h, clean_black_buf(mask.into_raw(), w, h, &p)).unwrap()
    }

    fn disk(mask: &mut Mask, cx: i32, cy: i32, r: i32, v: u8) {
        for y in (cy - r)..=(cy + r) {
            for x in (cx - r)..=(cx + r) {
                if (x - cx).pow(2) + (y - cy).pow(2) <= r * r {
                    mask.put_pixel(x as u32, y as u32, Luma([v]));
                }
            }
        }
    }

    fn count(m: &Mask) -> usize {
        m.as_raw().iter().filter(|&&v| v > 0).count()
    }

    fn random_mask(w: u32, h: u32, seed: u64, density: u64) -> Mask {
        GrayImage::from_raw(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    if hash_u64(u64::from(i % w / 2), u64::from(i / w / 2), seed).is_multiple_of(density) {
                        255
                    } else {
                        0
                    }
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn xdog_finds_square() {
        let mut lab = LabPlane::new(200, 200, [95.0, 0.0, 0.0]);
        for y in 60..140 {
            for x in 60..140 {
                lab.set(x, y, [60.0, 0.0, 0.0]);
            }
        }
        let f = xdog_field(&lab, &XdogParams::default());
        let m = threshold(&f, 0.15);
        let ink = |x: i32, y: i32| m.get_pixel(x as u32, y as u32)[0] > 0;
        let near = |x: i32, y: i32| {
            (-3..=3).any(|dy| {
                (-3..=3).any(|dx| {
                    let (px, py) = (x + dx, y + dy);
                    (0..200).contains(&px) && (0..200).contains(&py) && ink(px, py)
                })
            })
        };
        let mut hit = 0;
        let mut total = 0;
        for t in 60..140 {
            for (x, y) in [(t, 60), (t, 139), (60, t), (139, t)] {
                total += 1;
                hit += near(x, y) as usize;
            }
        }
        assert!(hit as f32 / total as f32 >= 0.9, "{hit}/{total}");
        for y in 0..200i32 {
            for x in 0..200i32 {
                if ink(x, y) {
                    let dx = (x - 60).abs().min((x - 139).abs());
                    let dy = (y - 60).abs().min((y - 139).abs());
                    let inside = (60..140).contains(&x) && (60..140).contains(&y);
                    let d = if inside {
                        dx.min(dy)
                    } else {
                        let ex = (60 - x).max(x - 139).max(0);
                        let ey = (60 - y).max(y - 139).max(0);
                        ((ex * ex + ey * ey) as f32).sqrt() as i32
                    };
                    assert!(d < 10, "stray ink at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn gaussian_is_bit_identical_to_clamped_reference() {
        for (w, h, sigma) in [(301u32, 17u32, 1.2f32), (5, 40, 1.92), (1, 9, 1.2), (9, 1, 1.2)] {
            let data: Vec<f32> = (0..w * h).map(|i| (hash_u64(u64::from(i), 5, 6) % 1000) as f32 / 1000.0).collect();
            let src = Plane::from_vec(w, h, data);
            let a = gaussian(&src, sigma);
            let b = reference::gaussian(&src, sigma);
            assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()), "{w}x{h}");
        }
    }

    #[test]
    fn small_components() {
        let mut m = GrayImage::new(100, 100);
        for y in 10..12 {
            for x in 10..13 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        m.put_pixel(14, 10, Luma([255]));
        for y in 40..60 {
            for x in 40..65 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        remove_small_components(&mut m, 20);
        assert_eq!(m.get_pixel(10, 10)[0], 0);
        assert_eq!(m.get_pixel(14, 10)[0], 0);
        assert_eq!(count(&m), 500);
    }

    #[test]
    fn tiny_masks_do_not_panic() {
        // A 1x1 all-ink mask must not panic.
        let mut m = GrayImage::from_pixel(1, 1, Luma([255]));
        remove_small_components(&mut m, 5);
        assert_eq!(m.get_pixel(0, 0)[0], 0);
        let mut m = GrayImage::from_pixel(1, 1, Luma([255]));
        remove_small_components(&mut m, 1);
        fill_small_holes(&mut m, 5);
        assert_eq!(m.get_pixel(0, 0)[0], 255);
        let mut m = GrayImage::from_pixel(1, 1, Luma([0]));
        fill_small_holes(&mut m, 5);
        assert_eq!(m.get_pixel(0, 0)[0], 0);
        let mut c = GrayImage::new(64, 64);
        for (i, p) in c.pixels_mut().enumerate() {
            p.0[0] = if (i % 64 + i / 64) % 2 == 0 { 255 } else { 0 };
        }
        fill_small_holes(&mut c, 4);
        assert_eq!(c.get_pixel(5, 5)[0], 255, "interior 1 px holes are filled");
        assert_eq!(c.get_pixel(1, 0)[0], 0, "border holes are kept");
        remove_small_components(&mut c, 4);
        assert!(count(&c) > 3000);
    }

    #[test]
    fn component_filters_match_imageproc_versions() {
        for seed in 0..12 {
            for density in [2u64, 3, 5] {
                let base = random_mask(53, 41, seed, density);
                let (mut a, mut b) = (base.clone(), base.clone());
                reference::remove_small_components(&mut a, 9, true);
                remove_small_components(&mut b, 9);
                assert_eq!(a, b, "remove seed {seed} density {density}");
                let (mut a, mut b) = (base.clone(), base);
                reference::fill_small_holes(&mut a, 9);
                fill_small_holes(&mut b, 9);
                assert_eq!(a, b, "fill seed {seed} density {density}");
            }
        }
    }

    #[test]
    fn spotted_blacks_cleanup() {
        let mut m = GrayImage::new(600, 600);
        disk(&mut m, 300, 300, 100, 255);
        let before = count(&m);
        disk(&mut m, 300, 300, 4, 0);
        disk(&mut m, 50, 50, 3, 255);
        let out = spotted_blacks(m);
        assert!(count(&out) as f32 >= 0.9 * before as f32);
        assert_eq!(out.get_pixel(50, 50)[0], 0);
        assert_eq!(out.get_pixel(300, 300)[0], 255);
    }

    #[test]
    fn work_masks_match_wrappers_at_f1() {
        let mut lab = LabPlane::new(300, 200, [90.0, 0.0, 0.0]);
        for y in 50..150 {
            for x in 60..200 {
                lab.set(x, y, [10.0, 0.0, 0.0]);
            }
        }
        let g = MaskScale::new(300, 200, 300);
        let boost = Boost::identity();
        let field = black_field(&lab, &boost);
        let work = black_mask_work(&field, 0.25, &BlackParams::scaled(&g));
        let full = spotted_blacks(threshold(&field, 0.25));
        assert_eq!(work, full.into_raw());
        assert_eq!(work[100 * 300 + 100], 255);
        assert_eq!(work[10 * 300 + 10], 0);
        let x = xdog_field(&lab, &XdogParams::default());
        let lw = line_mask_work(&x, 0.1, &MaskParams::scaled(&g, 1.0, 1.0));
        let lf = clean_lines(threshold(&x, 0.1), 1.0, 1.0);
        assert_eq!(lw, lf.into_raw());
    }

    #[test]
    fn scaled_params_follow_the_reference_table() {
        let g = MaskScale::new(4032, 3024, 2016);
        assert!((g.s - 1.008).abs() < 1e-3 && (g.f - 0.5).abs() < 1e-6);
        let p = MaskParams::scaled(&g, 1.0, 1.0);
        assert_eq!(p.min_area, (120.0 * 1.008f32 * 1.008 * 0.25) as u32);
        assert_eq!(p.close_r, 1.5);
        let tiny = MaskScale::new(4032, 3024, 1000);
        assert_eq!(MaskParams::scaled(&tiny, 1.0, 1.0).close_r, 0.0);
        assert_eq!(p.dilate_r, 1.5);
        let b = BlackParams::scaled(&g);
        assert!((b.open_r - 1.008).abs() < 1e-3 && (b.close_r - 1.512).abs() < 1e-3);
        assert_eq!(b.area, (4032.0f32 * 3024.0 * 0.0005 * 0.25) as u32);
        let full = MaskScale::new(4032, 3024, 4032);
        assert_eq!(MaskParams::scaled(&full, 1.0, 1.0).close_r, 1.5);
        assert_eq!(MaskParams::scaled(&full, 1.0, 0.0).dilate_r, 0.0);
    }

    #[test]
    fn clean_lines_runs() {
        let mut m = GrayImage::new(400, 400);
        for x in 50..350 {
            m.put_pixel(x, 200, Luma([255]));
        }
        m.put_pixel(10, 10, Luma([255]));
        let column = |m: &Mask, x: u32| (0..400).filter(|&y| m.get_pixel(x, y)[0] > 0).count();
        let out = clean_lines(m.clone(), 1.0, 1.0);
        assert_eq!(out.get_pixel(10, 10)[0], 0);
        assert!((50..350).all(|x| out.get_pixel(x, 200)[0] == 255), "line is continuous");
        assert!((50..350).all(|x| column(&out, x) == 1), "line stays one pixel thick");
        let thick = clean_lines(m, 1.0, 5.0);
        assert_eq!(column(&thick, 200), 5);
    }

    #[test]
    #[ignore = "benchmark; cargo test --release -- --ignored --nocapture"]
    fn bench_masks() {
        use crate::testutil::time;
        for (w, h) in [(2016u32, 1512u32), (4032, 3024)] {
            let data: Vec<[f32; 3]> = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32, (i / w) as f32);
                    [
                        50.0 + 45.0 * (x * 0.013).sin() * (y * 0.011).cos() + 8.0 * (x * 0.9).sin() * (y * 0.7).cos(),
                        0.0,
                        0.0,
                    ]
                })
                .collect();
            let lab = Plane::from_vec(w, h, data);
            let g = MaskScale::new(w, h, w);
            let x = time(&format!("{w}x{h} xdog"), || xdog_field(&lab, &XdogParams::default()));
            let lm =
                time(&format!("{w}x{h} line_mask_work"), || line_mask_work(&x, 0.1, &MaskParams::scaled(&g, 1.0, 1.0)));
            let f = black_field(&lab, &Boost::identity());
            let bm = time(&format!("{w}x{h} black_mask_work"), || black_mask_work(&f, 0.4, &BlackParams::scaled(&g)));
            let rl = time(&format!("{w}x{h} label_runs"), || crate::cc::label_runs(&bm, w, h, true, Some(0)));
            time(&format!("{w}x{h} disk_open r2"), || crate::morph::disk_open(&bm, w, h, 2.0));
            eprintln!(
                "{} line px, {} black px, {} labels",
                lm.iter().filter(|&&v| v > 0).count(),
                bm.iter().filter(|&&v| v > 0).count(),
                rl.n_labels
            );
        }
    }
}
