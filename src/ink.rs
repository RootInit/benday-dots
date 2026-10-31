use crate::types::InkSet;
use palette::white_point::D65;
use palette::{FromColor, Lab, LinSrgb, Srgb};

pub fn srgb8_to_lin(c: [u8; 3]) -> [f32; 3] {
    let l = Srgb::new(c[0], c[1], c[2]).into_format::<f32>().into_linear();
    [l.red, l.green, l.blue]
}

pub fn lin_to_srgb8(c: [f32; 3]) -> [u8; 3] {
    let lin = LinSrgb::new(c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0));
    let s = Srgb::<f32>::from_linear(lin);
    let q = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    [q(s.red), q(s.green), q(s.blue)]
}

pub fn lin_to_lab(c: [f32; 3]) -> [f32; 3] {
    let lab = Lab::<D65, f32>::from_color(LinSrgb::new(c[0], c[1], c[2]));
    [lab.l, lab.a, lab.b]
}

/// May be out of gamut; caller clamps.
fn lab_to_lin(lab: [f32; 3]) -> [f32; 3] {
    let l = LinSrgb::<f32>::from_color(Lab::<D65, f32>::new(lab[0], lab[1], lab[2]));
    [l.red, l.green, l.blue]
}

pub fn srgb8_to_lab(c: [u8; 3]) -> [f32; 3] {
    lin_to_lab(srgb8_to_lin(c))
}

pub fn lab_to_srgb8(lab: [f32; 3]) -> [u8; 3] {
    lin_to_srgb8(lab_to_lin(lab))
}

/// sqrt((l_weight * dL)^2 + da^2 + db^2)
pub fn delta_e(a: [f32; 3], b: [f32; 3], l_weight: f32) -> f32 {
    let dl = l_weight * (a[0] - b[0]);
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// Yellowed newsprint.
pub const PAPER_NEWSPRINT: [u8; 3] = [0xF2, 0xE8, 0xCF];
/// Pure white.
pub const PAPER_WHITE: [u8; 3] = [0xFF, 0xFF, 0xFF];
/// Bright cover stock.
pub const PAPER_COVER: [u8; 3] = [0xFA, 0xF7, 0xF0];

impl InkSet {
    /// Slightly muted Silver-Age inks on newsprint.
    pub fn aged() -> Self {
        InkSet {
            cyan: [0x01, 0xA0, 0xE7],
            magenta: [0xFE, 0x32, 0x8B],
            yellow: [0xFA, 0xF7, 0x00],
            black: [0x04, 0x04, 0x04],
            paper: PAPER_NEWSPRINT,
        }
    }

    /// Saturated modern process inks on newsprint.
    pub fn process() -> Self {
        InkSet {
            cyan: [0x00, 0xAE, 0xEF],
            magenta: [0xEC, 0x00, 0x8C],
            yellow: [0xFF, 0xF2, 0x00],
            black: [0x04, 0x04, 0x04],
            paper: PAPER_NEWSPRINT,
        }
    }

    /// The same inks on a different paper.
    #[must_use]
    pub fn with_paper(self, paper: [u8; 3]) -> Self {
        InkSet { paper, ..self }
    }
}

const SRGB_LUT_N: usize = 1 << 14;

/// Linear to sRGB8 by table, within 1 LSB of `lin_to_srgb8`; no transcendental per pixel.
pub struct SrgbLut(Vec<u8>);

impl SrgbLut {
    /// Build the 16384-entry table.
    pub fn new() -> Self {
        SrgbLut((0..SRGB_LUT_N).map(|i| lin_to_srgb8([(i as f32 + 0.5) / SRGB_LUT_N as f32; 3])[0]).collect())
    }

    /// sRGB8 of linear value `lin`, saturating outside 0..1.
    #[inline(always)]
    pub fn get(&self, lin: f32) -> u8 {
        self.0[((lin * SRGB_LUT_N as f32) as i32).clamp(0, SRGB_LUT_N as i32 - 1) as usize]
    }
}

impl Default for SrgbLut {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear-light multiplicative Neugebauer model.
#[derive(Clone, Copy, Debug)]
pub struct InkModel {
    pub paper_lin: [f32; 3],
    /// Transmittance of C, M, Y.
    pub trans: [[f32; 3]; 3],
    pub trans_k: [f32; 3],
}

impl InkModel {
    /// Swatches are "ink on white", so they are used directly as transmittance over the chosen paper.
    pub fn new(inks: &InkSet) -> Self {
        InkModel {
            paper_lin: srgb8_to_lin(inks.paper),
            trans: [srgb8_to_lin(inks.cyan), srgb8_to_lin(inks.magenta), srgb8_to_lin(inks.yellow)],
            trans_k: srgb8_to_lin(inks.black),
        }
    }

    /// lin = paper_lin * paper_gain (.) prod_i (1 - cov_i (1 - T_i)); cov in 0..1.
    pub fn mix(&self, cov: [f32; 3], paper_gain: f32) -> [f32; 3] {
        let mut out = [0.0; 3];
        for (ch, o) in out.iter_mut().enumerate() {
            let mut v = self.paper_lin[ch] * paper_gain;
            for (c, t) in cov.iter().zip(&self.trans) {
                v *= 1.0 - c * (1.0 - t[ch]);
            }
            *o = v;
        }
        out
    }

    /// `[ink][channel][coverage as u8]`: the factor `mix` multiplies in for that quantised coverage, so
    /// the product of three table lookups equals `mix` exactly.
    pub fn mix_tables(&self) -> [[[f32; 256]; 3]; 3] {
        std::array::from_fn(|i| {
            std::array::from_fn(|ch| std::array::from_fn(|c| 1.0 - (c as f32 / 255.0) * (1.0 - self.trans[i][ch])))
        })
    }

    /// K over everything: lerp(lin, lin (.) T_K, alpha).
    pub fn apply_k(&self, lin: [f32; 3], alpha: f32) -> [f32; 3] {
        let mut out = [0.0; 3];
        for ch in 0..3 {
            let inked = lin[ch] * self.trans_k[ch];
            out[ch] = lin[ch] + (inked - lin[ch]) * alpha;
        }
        out
    }
}

/// Mean area coverage the round-dot screen produces for nominal tint t at pitch p and dot gain g.
/// Dots below 0.5, inverted holes above; the gain never applies at the exact endpoints.
pub fn effective_coverage(t: f32, pitch: f32, dot_gain: f32) -> f32 {
    use std::f32::consts::PI;
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let half = pitch * 0.5;
    if t <= 0.5 {
        let r = (pitch * (t / PI).sqrt() + dot_gain).clamp(0.0, half);
        PI * r * r / (pitch * pitch)
    } else {
        let rh = (pitch * ((1.0 - t) / PI).sqrt() - dot_gain).clamp(0.0, half);
        1.0 - PI * rh * rh / (pitch * pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::LEVELS;
    use crate::types::InkSet;

    const SWATCHES: [[u8; 3]; 6] =
        [[0, 0, 0], [255, 255, 255], [0x01, 0xA0, 0xE7], [0xFE, 0x32, 0x8B], [0xFA, 0xF7, 0x00], [0xF2, 0xE8, 0xCF]];

    #[test]
    fn srgb_lin_lab_round_trip() {
        for c in SWATCHES {
            assert_eq!(lin_to_srgb8(srgb8_to_lin(c)), c);
            assert_eq!(lab_to_srgb8(srgb8_to_lab(c)), c);
        }
    }

    #[test]
    fn lab_of_white() {
        let l = srgb8_to_lab([255, 255, 255]);
        assert!((l[0] - 100.0).abs() < 0.1 && l[1].abs() < 0.1 && l[2].abs() < 0.1, "{l:?}");
    }

    #[test]
    fn delta_e_weights_lightness() {
        assert!((delta_e([50.0, 0.0, 0.0], [52.0, 0.0, 0.0], 1.5) - 3.0).abs() < 1e-5);
        assert!((delta_e([50.0, 0.0, 0.0], [50.0, 3.0, 4.0], 1.5) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn mix_zero_is_paper() {
        let m = InkModel::new(&InkSet::aged());
        assert_eq!(m.mix([0.0; 3], 1.0), m.paper_lin);
    }

    #[test]
    fn solid_inks_on_white_match_swatches() {
        let inks = InkSet::aged().with_paper(PAPER_WHITE);
        let m = InkModel::new(&inks);
        let swatches = [inks.cyan, inks.magenta, inks.yellow];
        for (i, sw) in swatches.iter().enumerate() {
            let mut cov = [0.0; 3];
            cov[i] = 1.0;
            let got = lin_to_srgb8(m.mix(cov, 1.0));
            for ch in 0..3 {
                assert!((got[ch] as i32 - sw[ch] as i32).abs() <= 1, "ink {i}: {got:?} vs {sw:?}");
            }
        }
    }

    #[test]
    fn overprint_is_darker() {
        let m = InkModel::new(&InkSet::aged());
        let all = lin_to_lab(m.mix([1.0; 3], 1.0))[0];
        for i in 0..3 {
            let mut cov = [0.0; 3];
            cov[i] = 1.0;
            assert!(all < lin_to_lab(m.mix(cov, 1.0))[0]);
        }
    }

    #[test]
    fn apply_k_endpoints() {
        let m = InkModel::new(&InkSet::aged());
        let lin = [0.5, 0.4, 0.3];
        assert_eq!(m.apply_k(lin, 0.0), lin);
        let k = m.apply_k(lin, 1.0);
        for ch in 0..3 {
            assert!((k[ch] - lin[ch] * m.trans_k[ch]).abs() < 1e-6);
        }
    }

    #[test]
    fn coverage_identity_without_gain() {
        for p in [6.0, 10.0, 27.0] {
            for i in 0..=20 {
                let t = i as f32 * 0.05;
                assert!((effective_coverage(t, p, 0.0) - t).abs() < 1e-6, "t={t} p={p}");
            }
        }
    }

    #[test]
    fn coverage_endpoints_exact() {
        for g in [-2.0, -0.5, 0.0, 0.5, 2.0] {
            assert_eq!(effective_coverage(0.0, 10.0, g), 0.0);
            assert_eq!(effective_coverage(1.0, 10.0, g), 1.0);
        }
    }

    #[test]
    fn coverage_monotone() {
        for g in [-0.5f32, 0.0, 0.5] {
            for p in [6.0f32, 10.0, 27.0] {
                let ts: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
                let cov: Vec<f32> = ts.iter().map(|&t| effective_coverage(t, p, g)).collect();
                for i in 1..ts.len() {
                    let across_seam = ts[i - 1] <= 0.5 && ts[i] > 0.5;
                    if g == 0.0 || !across_seam {
                        assert!(cov[i] >= cov[i - 1] - 1e-6, "g={g} p={p} t={}", ts[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn levels_monotone_for_any_gain() {
        for g in [-1.0f32, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0] {
            for p in [6.0f32, 10.0, 27.0] {
                let c: Vec<f32> = LEVELS.iter().map(|&t| effective_coverage(t, p, g)).collect();
                assert!(c.windows(2).all(|w| w[1] >= w[0]), "g={g} p={p} {c:?}");
            }
        }
    }

    #[test]
    fn mix_tables_reproduce_mix_exactly() {
        let m = InkModel::new(&InkSet::aged());
        let t = m.mix_tables();
        for c0 in (0..=255usize).step_by(5) {
            for c1 in (0..=255usize).step_by(7) {
                for c2 in (0..=255usize).step_by(11) {
                    let want = m.mix([c0 as f32 / 255.0, c1 as f32 / 255.0, c2 as f32 / 255.0], 1.0);
                    for ch in 0..3 {
                        let got = m.paper_lin[ch] * t[0][ch][c0] * t[1][ch][c1] * t[2][ch][c2];
                        assert_eq!(got.to_bits(), want[ch].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn srgb_lut_within_one_lsb() {
        let l = SrgbLut::new();
        for i in 0..=100_000 {
            let v = i as f32 / 100_000.0;
            let (a, b) = (l.get(v) as i32, lin_to_srgb8([v; 3])[0] as i32);
            assert!((a - b).abs() <= 1, "{v} {a} {b}");
        }
        assert_eq!((l.get(-1.0), l.get(2.0)), (0, 255));
    }
}
