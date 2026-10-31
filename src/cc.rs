//! Connected components on run-length encoded rows.
//!
//! Runs are extracted in parallel; the union-find then works on runs instead of pixels, so it is cheap even
//! on 12 MP maps. Unlike imageproc's labeller it also handles maps where every pixel is its own component.
use rayon::prelude::*;

/// A horizontal run `[x0, x1)` of equal-valued pixels.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// First pixel of the run.
    pub x0: u32,
    /// One past the last pixel of the run.
    pub x1: u32,
    /// The pixel value shared by the run.
    pub val: u8,
    /// Component label, 1.. (background runs are not stored, so 0 never appears here).
    pub label: u32,
}

/// Runs of every non-background pixel plus their component labels.
pub struct RunLabels {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// `row_start[y]..row_start[y + 1]` indexes `runs` for row `y`; length is `height + 1`.
    pub row_start: Vec<u32>,
    /// All runs in raster order.
    pub runs: Vec<Run>,
    /// Number of labels including background 0.
    pub n_labels: u32,
}

impl RunLabels {
    /// The runs of row `y`, left to right.
    pub fn row(&self, y: usize) -> &[Run] {
        &self.runs[self.row_start[y] as usize..self.row_start[y + 1] as usize]
    }

    /// Pixel count per label; index 0 (background) is left at 0.
    pub fn areas(&self) -> Vec<u32> {
        let mut a = vec![0u32; self.n_labels as usize];
        for r in &self.runs {
            a[r.label as usize] += r.x1 - r.x0;
        }
        a
    }

    /// Whether each label has a pixel on the image border; index 0 is `false`.
    pub fn touches_border(&self) -> Vec<bool> {
        let (w, h) = (self.width, self.height as usize);
        let mut border = vec![false; self.n_labels as usize];
        for y in 0..h {
            let edge_row = y == 0 || y + 1 == h;
            for r in self.row(y) {
                if edge_row || r.x0 == 0 || r.x1 == w {
                    border[r.label as usize] = true;
                }
            }
        }
        border
    }

    /// Per-pixel label map. Only for tests and debugging.
    pub fn to_pixels(&self) -> Vec<u32> {
        let w = self.width as usize;
        let mut out = vec![0u32; w * self.height as usize];
        if w == 0 {
            return out;
        }
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for r in self.row(y) {
                row[r.x0 as usize..r.x1 as usize].fill(r.label);
            }
        });
        out
    }
}

fn find(p: &mut [u32], mut i: u32) -> u32 {
    while p[i as usize] != i {
        let g = p[p[i as usize] as usize];
        p[i as usize] = g;
        i = g;
    }
    i
}

/// Labels the connected components of equal-valued pixels (4- or 8-neighbourhood). Pixels equal to
/// `background` are label 0 and have no runs. Labels run 1.. in raster order of each component's first pixel.
pub fn label_runs(data: &[u8], width: u32, height: u32, eight: bool, background: Option<u8>) -> RunLabels {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(data.len(), w * h, "label_runs: data length mismatch");
    let rows: Vec<Vec<Run>> = (0..h)
        .into_par_iter()
        .map(|y| {
            let line = &data[y * w..(y + 1) * w];
            let mut v = Vec::new();
            let mut x = 0;
            while x < w {
                let c = line[x];
                let s = x;
                while x < w && line[x] == c {
                    x += 1;
                }
                if Some(c) != background {
                    v.push(Run { x0: s as u32, x1: x as u32, val: c, label: 0 });
                }
            }
            v
        })
        .collect();
    let mut row_start = Vec::with_capacity(h + 1);
    let mut n = 0u32;
    for r in &rows {
        row_start.push(n);
        n += r.len() as u32;
    }
    row_start.push(n);
    let mut runs: Vec<Run> = rows.into_iter().flatten().collect();
    // The root of each set is its smallest run index, i.e. the raster-first run, which gives the label order.
    let mut parent: Vec<u32> = (0..runs.len() as u32).collect();
    let slack = u32::from(eight);
    for y in 1..h {
        let (a0, a1) = (row_start[y - 1] as usize, row_start[y] as usize);
        let (b0, b1) = (row_start[y] as usize, row_start[y + 1] as usize);
        let mut i = a0;
        for j in b0..b1 {
            let rb = runs[j];
            let lo = rb.x0.saturating_sub(slack);
            let hi = rb.x1 + slack;
            while i < a1 && runs[i].x1 <= lo {
                i += 1;
            }
            let mut k = i;
            while k < a1 && runs[k].x0 < hi {
                if runs[k].val == rb.val {
                    let (ra, rj) = (find(&mut parent, k as u32), find(&mut parent, j as u32));
                    if ra != rj {
                        parent[ra.max(rj) as usize] = ra.min(rj);
                    }
                }
                k += 1;
            }
        }
    }
    let mut next = 1u32;
    for i in 0..runs.len() {
        let r = find(&mut parent, i as u32) as usize;
        if r == i {
            runs[i].label = next;
            next += 1;
        } else {
            runs[i].label = runs[r].label;
        }
    }
    RunLabels { width, height, row_start, runs, n_labels: next }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::hash_u64;
    use image::{GrayImage, Luma};
    use imageproc::region_labelling::{Connectivity, connected_components};
    use std::collections::HashMap;

    fn rnd(n: usize, seed: u64, k: u64) -> Vec<u8> {
        (0..n).map(|i| (hash_u64(i as u64 / 3, i as u64 % 7, seed) % k) as u8).collect()
    }

    /// `a` and `b` describe the same partition, with 0 mapping to 0 only.
    fn assert_same_partition(a: &[u32], b: &[u32], what: &str) {
        let mut fwd = HashMap::new();
        let mut rev = HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            assert_eq!(x == 0, y == 0, "{what}: background differs");
            assert_eq!(*fwd.entry(x).or_insert(y), y, "{what}");
            assert_eq!(*rev.entry(y).or_insert(x), x, "{what}");
        }
    }

    fn flood_fill(d: &[u8], w: usize, h: usize, eight: bool, bg: Option<u8>) -> Vec<u32> {
        let mut lab = vec![0u32; w * h];
        let mut next = 0;
        for s in 0..w * h {
            if lab[s] != 0 || Some(d[s]) == bg {
                continue;
            }
            next += 1;
            lab[s] = next;
            let mut stack = vec![s];
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % w) as i64, (i / w) as i64);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if (dx == 0 && dy == 0) || (!eight && dx != 0 && dy != 0) {
                            continue;
                        }
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if lab[j] == 0 && d[j] == d[s] {
                            lab[j] = next;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        lab
    }

    #[test]
    fn matches_imageproc_up_to_relabelling() {
        for seed in 0..20 {
            let (w, h) = (37u32, 29u32);
            let mut d = rnd((w * h) as usize, seed, 3);
            for v in &mut d {
                *v *= 100;
            }
            for (eight, bg) in [(false, None), (true, Some(0u8)), (false, Some(200u8)), (true, Some(255u8))] {
                let rl = label_runs(&d, w, h, eight, bg);
                let img = GrayImage::from_raw(w, h, d.clone()).unwrap();
                let conn = if eight { Connectivity::Eight } else { Connectivity::Four };
                let want = connected_components(&img, conn, Luma([bg.unwrap_or(255)]));
                let got = rl.to_pixels();
                assert_same_partition(&got, want.as_raw(), &format!("seed {seed} eight {eight} bg {bg:?}"));
                let max = *got.iter().max().unwrap();
                assert_eq!(max + 1, rl.n_labels);
            }
        }
    }

    #[test]
    fn matches_flood_fill_on_odd_shapes() {
        for (w, h) in [(1u32, 1u32), (1, 13), (13, 1), (2, 1), (1, 2), (2, 2), (5, 3)] {
            for seed in 0..10 {
                let d = rnd((w * h) as usize, seed, 3);
                for eight in [false, true] {
                    for bg in [None, Some(0u8), Some(2)] {
                        let got = label_runs(&d, w, h, eight, bg).to_pixels();
                        let want = flood_fill(&d, w as usize, h as usize, eight, bg);
                        assert_eq!(got, want, "{w}x{h} seed {seed} eight {eight} bg {bg:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn every_pixel_its_own_region() {
        let rl = label_runs(&[7], 1, 1, false, None);
        assert_eq!((rl.n_labels, rl.to_pixels()), (2, vec![1]));
        let rl = label_runs(&[1, 2], 2, 1, false, None);
        assert_eq!((rl.n_labels, rl.to_pixels()), (3, vec![1, 2]));
        let (w, h) = (64usize, 64usize);
        let d: Vec<u8> = (0..w * h).map(|i| ((i % w + i / w) % 2) as u8).collect();
        let four = label_runs(&d, w as u32, h as u32, false, None);
        assert_eq!(four.n_labels, (w * h) as u32 + 1);
        assert_eq!(four.areas()[1..].iter().sum::<u32>(), (w * h) as u32);
        let eight = label_runs(&d, w as u32, h as u32, true, None);
        assert_eq!(eight.n_labels, 3);
    }

    #[test]
    fn empty_and_border() {
        assert_eq!(label_runs(&[], 0, 0, false, None).n_labels, 1);
        assert_eq!(label_runs(&[], 0, 4, true, None).to_pixels().len(), 0);
        let mut d = vec![0u8; 25];
        d[12] = 1;
        d[0] = 1;
        let rl = label_runs(&d, 5, 5, false, Some(0));
        assert_eq!(rl.touches_border(), vec![false, true, false]);
        assert_eq!(rl.areas(), vec![0, 1, 1]);
    }
}
