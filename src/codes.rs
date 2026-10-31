use crate::ink::{InkModel, delta_e, effective_coverage, lin_to_lab, lin_to_srgb8};
use crate::types::Code;
use rayon::prelude::*;

/// Nominal tint levels per level index 0..=3 (Silver-Age 0/25/50/100).
pub const LEVELS: [f32; 4] = [0.0, 0.25, 0.5, 1.0];

/// Pop palette: paper, Y, R, YR, B, RB, YB (green) solid, plus Y3, R3, B3 and Y3B3 (green) tints, and
/// full overprint (63, prints as black) as the dark fill (code = c*16 + m*4 + y).
pub const POP_CODES: [u8; 12] = [0, 3, 12, 15, 48, 60, 51, 2, 8, 32, 34, 63];

/// What each of the 64 codes looks like when printed.
pub struct RenderedPalette {
    /// `[c,m,y]` nominal, indexed by Code::index; len 64.
    pub tints: Vec<[f32; 3]>,
    /// Predicted mean appearance in Lab.
    pub lab: Vec<[f32; 3]>,
    /// sRGB8 of each code, for previews.
    pub srgb: Vec<[u8; 3]>,
    /// Lab chroma of each code, kept for the hue penalty of [`RenderedPalette::nearest_hue`].
    pub chroma: Vec<f32>,
}

impl RenderedPalette {
    /// Predict all 64 codes for the given inks, tint levels, dot pitch and dot gain.
    pub fn build(model: &InkModel, levels: &[f32; 4], pitch: f32, dot_gain: f32) -> Self {
        let mut tints = Vec::with_capacity(64);
        let mut lab = Vec::with_capacity(64);
        let mut srgb = Vec::with_capacity(64);
        for i in 0..64u8 {
            let t = Code::from_index(i).tints(levels);
            let a = t.map(|t| effective_coverage(t, pitch, dot_gain));
            let l = model.mix(a, 1.0);
            tints.push(t);
            lab.push(lin_to_lab(l));
            srgb.push(lin_to_srgb8(l));
        }
        let chroma = lab.iter().map(|q| q[1].hypot(q[2])).collect();
        RenderedPalette { tints, lab, srgb, chroma }
    }

    /// The code in `allowed` closest to `lab`. `allowed` must not be empty.
    pub fn nearest(&self, lab: [f32; 3], l_weight: f32, allowed: &[u8]) -> u8 {
        self.nearest_hue(lab, l_weight, 0.0, allowed)
    }

    /// Like `nearest`, plus a penalty `hue_weight * min(C_pixel, C_code) * (1 - cos(hue difference))` so a coloured
    /// pixel does not snap to a code of the wrong hue just because the lightness matches.
    pub fn nearest_hue(&self, lab: [f32; 3], l_weight: f32, hue_weight: f32, allowed: &[u8]) -> u8 {
        debug_assert!(!allowed.is_empty(), "nearest_hue needs at least one allowed code");
        let mut best = allowed[0];
        let mut best_d = f32::INFINITY;
        let cp = lab[1].hypot(lab[2]);
        for &i in allowed {
            let q = self.lab[i as usize];
            let mut d = delta_e(lab, q, l_weight);
            if hue_weight > 0.0 {
                let cq = self.chroma[i as usize];
                if cp > 1e-3 && cq > 1e-3 {
                    let cos = (lab[1] * q[1] + lab[2] * q[2]) / (cp * cq);
                    d += hue_weight * cp.min(cq) * (1.0 - cos);
                }
            }
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

const N: usize = 64;
const A_MIN: f32 = -128.0;
const A_RANGE: f32 = 256.0;
const L_RANGE: f32 = 100.0;

/// Nearest allowed code for every cell of a 64x64x64 Lab grid.
pub struct PaletteLut {
    lut: Vec<u8>,
}

impl PaletteLut {
    /// Table for plain lightness-weighted distance.
    pub fn new(pal: &RenderedPalette, l_weight: f32, allowed: &[u8]) -> Self {
        Self::with_hue(pal, l_weight, 0.0, allowed)
    }

    /// Table with the hue penalty of [`RenderedPalette::nearest_hue`]. `allowed` must not be empty.
    pub fn with_hue(pal: &RenderedPalette, l_weight: f32, hue_weight: f32, allowed: &[u8]) -> Self {
        let mut lut = vec![0u8; N * N * N];
        lut.par_chunks_mut(N * N).enumerate().for_each(|(li, slab)| {
            let l = (li as f32 + 0.5) * L_RANGE / N as f32;
            for ia in 0..N {
                let a = A_MIN + (ia as f32 + 0.5) * A_RANGE / N as f32;
                for bi in 0..N {
                    let b = A_MIN + (bi as f32 + 0.5) * A_RANGE / N as f32;
                    slab[ia * N + bi] = pal.nearest_hue([l, a, b], l_weight, hue_weight, allowed);
                }
            }
        });
        PaletteLut { lut }
    }

    /// The code for a Lab colour (cells outside the grid clamp to its edge).
    #[inline]
    pub fn lookup(&self, lab: [f32; 3]) -> u8 {
        let cell =
            |v: f32, min: f32, range: f32| (((v - min) * (N as f32 / range)) as i32).clamp(0, N as i32 - 1) as usize;
        let l = cell(lab[0], 0.0, L_RANGE);
        let a = cell(lab[1], A_MIN, A_RANGE);
        let b = cell(lab[2], A_MIN, A_RANGE);
        self.lut[(l * N + a) * N + b]
    }
}

/// All 64 code numbers.
pub fn all_codes() -> Vec<u8> {
    (0..64).collect()
}

/// Aged inks on `paper` and the 64-code palette they print at pitch 10, no dot gain.
#[cfg(test)]
pub(crate) fn test_palette(paper: [u8; 3]) -> (InkModel, RenderedPalette) {
    let model = InkModel::new(&crate::types::InkSet::aged().with_paper(paper));
    let pal = RenderedPalette::build(&model, &LEVELS, 10.0, 0.0);
    (model, pal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ink::{PAPER_NEWSPRINT, PAPER_WHITE};
    use crate::noise::hash01;
    use crate::types::InkSet;

    #[test]
    fn paper_and_solid_cyan() {
        let (_, pal) = test_palette(PAPER_NEWSPRINT);
        for (got, want) in pal.srgb[0].iter().zip(PAPER_NEWSPRINT) {
            assert!((i32::from(*got) - i32::from(want)).abs() <= 1);
        }
        let (_, pal) = test_palette(PAPER_WHITE);
        let idx = Code { c: 3, m: 0, y: 0 }.index() as usize;
        for (got, want) in pal.srgb[idx].iter().zip(InkSet::aged().cyan) {
            assert!((i32::from(*got) - i32::from(want)).abs() <= 1);
        }
    }

    #[test]
    fn lightness_decreases_with_level() {
        let (_, pal) = test_palette(PAPER_NEWSPRINT);
        let mut steps = 0;
        for i in 0..64u8 {
            let c = Code::from_index(i);
            for ink in 0..3 {
                let lv = [c.c, c.m, c.y][ink];
                if lv == 3 {
                    continue;
                }
                let mut n = [c.c, c.m, c.y];
                n[ink] += 1;
                let j = Code { c: n[0], m: n[1], y: n[2] }.index();
                assert!(pal.lab[j as usize][0] < pal.lab[i as usize][0], "code {i} ink {ink}");
                steps += 1;
            }
        }
        assert_eq!(steps, 144);
    }

    #[test]
    fn nearest_round_trip() {
        let (_, pal) = test_palette(PAPER_NEWSPRINT);
        let all = all_codes();
        for i in 0..64u8 {
            assert_eq!(pal.nearest(pal.lab[i as usize], 1.5, &all), i);
        }
    }

    #[test]
    fn lut_bounded_suboptimality() {
        let (_, pal) = test_palette(PAPER_NEWSPRINT);
        let all = all_codes();
        let lut = PaletteLut::new(&pal, 1.5, &all);
        let (mut n, mut agree, mut worst) = (0, 0, 0.0f32);
        let mut i = 0i64;
        while n < 10_000 {
            i += 1;
            let rgb = [hash01(i, 1, 0), hash01(i, 2, 0), hash01(i, 3, 0)];
            let srgb = rgb.map(|v| (v * 255.0) as u8);
            let p = crate::ink::srgb8_to_lab(srgb);
            let e = pal.nearest(p, 1.5, &all);
            let l = lut.lookup(p);
            agree += i32::from(e == l);
            let gap = delta_e(p, pal.lab[l as usize], 1.5) - delta_e(p, pal.lab[e as usize], 1.5);
            worst = worst.max(gap);
            assert!(gap < 6.5, "gap {gap}");
            n += 1;
        }
        eprintln!("lut exact agreement {:.1}%, worst gap {worst:.2}", agree as f32 / 100.0);
        let two = PaletteLut::new(&pal, 1.5, &[0, 63]);
        for k in 0..1000i64 {
            let p = [hash01(k, 4, 0) * 100.0, hash01(k, 5, 0) * 200.0 - 100.0, hash01(k, 6, 0) * 200.0 - 100.0];
            let v = two.lookup(p);
            assert!(v == 0 || v == 63);
        }
    }
}
