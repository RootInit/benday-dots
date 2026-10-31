//! Binary morphology with a disk structuring element `dx^2 + dy^2 <= r^2` (`r` may be fractional).
//!
//! Masks are `0` = off, non-zero = on; results are `0`/`255`. Cost is one horizontal distance pass plus
//! `2r + 1` row passes, which beats a whole-image distance transform for the small radii used here.
//! Out-of-image pixels are ignored, so neither operation reacts to the image border.
use rayon::prelude::*;

/// Per-row horizontal distance to the nearest pixel equal to `set`, capped at `cap`.
fn row_dist(line: &[u8], set: bool, cap: u16, out: &mut [u16]) {
    let w = line.len();
    let out = &mut out[..w];
    let mut d = cap;
    for (o, &p) in out.iter_mut().zip(line) {
        d = if (p != 0) == set { 0 } else { (d + 1).min(cap) };
        *o = d;
    }
    d = cap;
    for (o, &p) in out.iter_mut().zip(line).rev() {
        d = if (p != 0) == set { 0 } else { (d + 1).min(cap) };
        *o = (*o).min(d);
    }
}

/// Dilation of the pixels that equal `set` (true: dilate the mask; false: dilate the zeros, i.e. erode).
fn disk_op(src: &[u8], w: u32, h: u32, r: f32, set: bool) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    assert_eq!(src.len(), w * h, "morph: mask length mismatch");
    let (hit, miss) = if set { (255u8, 0u8) } else { (0u8, 255u8) };
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let r = if r.is_finite() { r.max(0.0) } else { 0.0 };
    let ri = (r.floor() as i64).min(i64::from(u16::MAX) - 1);
    let reach = ri.min(h as i64 - 1);
    let half_width = |dy: i64| ((r * r - (dy * dy) as f32).max(0.0)).sqrt().floor() as u16;
    let rows: Vec<(i64, u16)> = (-reach..=reach).map(|dy| (dy, half_width(dy))).collect();
    let cap = (ri + 1) as u16;
    let mut dist = vec![0u16; w * h];
    dist.par_chunks_mut(w).zip(src.par_chunks(w)).for_each(|(d, l)| row_dist(l, set, cap, d));
    let mut out = vec![miss; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for &(dy, lim) in &rows {
            let yy = y as i64 + dy;
            if yy < 0 || yy >= h as i64 {
                continue;
            }
            let d = &dist[yy as usize * w..(yy as usize + 1) * w];
            for (o, &v) in row.iter_mut().zip(d) {
                if v <= lim {
                    *o = hit;
                }
            }
        }
    });
    out
}

/// Dilation: a pixel is on if any on pixel lies within the disk.
pub fn disk_dilate(mask: &[u8], w: u32, h: u32, r: f32) -> Vec<u8> {
    disk_op(mask, w, h, r, true)
}

/// Erosion: a pixel stays on only if every in-image pixel within the disk is on.
pub fn disk_erode(mask: &[u8], w: u32, h: u32, r: f32) -> Vec<u8> {
    disk_op(mask, w, h, r, false)
}

/// Erosion then dilation: removes features narrower than the disk.
pub fn disk_open(mask: &[u8], w: u32, h: u32, r: f32) -> Vec<u8> {
    disk_dilate(&disk_erode(mask, w, h, r), w, h, r)
}

/// Dilation then erosion: fills gaps narrower than the disk.
pub fn disk_close(mask: &[u8], w: u32, h: u32, r: f32) -> Vec<u8> {
    disk_erode(&disk_dilate(mask, w, h, r), w, h, r)
}

/// 3x3 box mean with replicated borders, as two truncating 1-D means (horizontal, then vertical) like
/// `imageproc::filter::box_filter(img, 1, 1)`.
pub fn box3(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    assert_eq!(data.len(), w * h, "mask matches its dimensions");
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut mid = vec![0u8; w * h];
    mid.par_chunks_mut(w).zip(data.par_chunks(w)).for_each(|(o, line)| {
        for (x, o) in o.iter_mut().enumerate() {
            let (l, c, r) = (line[x.saturating_sub(1)], line[x], line[(x + 1).min(w - 1)]);
            *o = ((u32::from(l) + u32::from(c) + u32::from(r)) / 3) as u8;
        }
    });
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let up = &mid[y.saturating_sub(1) * w..][..w];
        let mid_row = &mid[y * w..][..w];
        let down = &mid[(y + 1).min(h - 1) * w..][..w];
        for (((o, &a), &b), &c) in o.iter_mut().zip(up).zip(mid_row).zip(down) {
            *o = ((u32::from(a) + u32::from(b) + u32::from(c)) / 3) as u8;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::hash_u64;

    #[test]
    fn box3_matches_imageproc() {
        for (w, h) in [(1u32, 1u32), (1, 7), (7, 1), (2, 2), (3, 3), (5, 9), (31, 17), (64, 65)] {
            for seed in 0..4u64 {
                let data: Vec<u8> = (0..w * h).map(|i| (hash_u64(i as u64, seed, 5) % 256) as u8).collect();
                let img = image::GrayImage::from_raw(w, h, data.clone()).unwrap();
                let want = imageproc::filter::box_filter(&img, 1, 1).into_raw();
                assert_eq!(box3(&data, w, h), want, "{w}x{h} seed {seed}");
            }
        }
        assert!(box3(&[], 0, 0).is_empty());
    }

    fn random_mask(w: u32, h: u32, seed: u64) -> Vec<u8> {
        (0..w * h).map(|i| if hash_u64(i as u64 / 3, i as u64 % 7, seed).is_multiple_of(4) { 255 } else { 0 }).collect()
    }

    /// Dilation of the pixels equal to `set` by the disk, ignoring out-of-image pixels.
    fn brute(d: &[u8], w: u32, h: u32, r: f32, set: bool) -> Vec<u8> {
        let ri = r.floor() as i64;
        let mut out = vec![0u8; d.len()];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut any = false;
                for dy in -ri..=ri {
                    for dx in -ri..=ri {
                        let (xx, yy) = (x + dx, y + dy);
                        if (dx * dx + dy * dy) as f32 <= r * r && xx >= 0 && yy >= 0 && xx < w as i64 && yy < h as i64 {
                            any |= (d[(yy * w as i64 + xx) as usize] != 0) == set;
                        }
                    }
                }
                out[(y * w as i64 + x) as usize] = if set == any { 255 } else { 0 };
            }
        }
        out
    }

    #[test]
    fn dilate_and_erode_match_brute_force() {
        for seed in 0..4 {
            let (w, h) = (40u32, 33u32);
            let d = random_mask(w, h, seed);
            for r in [0.0f32, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0] {
                assert_eq!(disk_dilate(&d, w, h, r), brute(&d, w, h, r, true), "dilate r {r}");
                assert_eq!(disk_erode(&d, w, h, r), brute(&d, w, h, r, false), "erode r {r}");
            }
        }
    }

    #[test]
    fn open_and_close_compose() {
        let (w, h) = (31u32, 27u32);
        let d = random_mask(w, h, 9);
        for r in [1.0f32, 1.5, 2.0, 3.0] {
            let open = brute(&brute(&d, w, h, r, false), w, h, r, true);
            let close = brute(&brute(&d, w, h, r, true), w, h, r, false);
            assert_eq!(disk_open(&d, w, h, r), open, "open r {r}");
            assert_eq!(disk_close(&d, w, h, r), close, "close r {r}");
        }
    }

    #[test]
    fn border_and_tiny_inputs() {
        assert!(disk_dilate(&[], 0, 0, 3.0).is_empty());
        assert_eq!(disk_dilate(&[255], 1, 1, 8.0), vec![255]);
        assert_eq!(disk_erode(&[255], 1, 1, 8.0), vec![255]);
        assert_eq!(disk_close(&[0, 255, 0], 3, 1, 1.0), vec![255; 3]);
        let mut d = vec![0u8; 5 * 5];
        d[0] = 1;
        let o = disk_dilate(&d, 5, 5, 1.0);
        assert_eq!(o.iter().map(|&v| usize::from(v == 255)).sum::<usize>(), 3);
    }
}
