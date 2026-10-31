use crate::ink::{InkModel, SrgbLut, lin_to_srgb8};
use crate::noise::{FbmBand, grain_seeds};
use crate::screen::BAND;
use crate::types::Coverage;
use image::{GrayImage, RgbImage};
use rayon::prelude::*;

/// Per-image constants of the mixing loop.
pub(crate) struct Mixer {
    paper: [f32; 3],
    trans_k: [f32; 3],
    fac: [[[f32; 256]; 3]; 3],
    srgb: SrgbLut,
}

impl Mixer {
    pub(crate) fn new(model: &InkModel) -> Self {
        Mixer { paper: model.paper_lin, trans_k: model.trans_k, fac: model.mix_tables(), srgb: SrgbLut::new() }
    }

    /// Subtractive mix of three quantised coverage rows on paper with grain `1 + paper_grain * (2n - 1)` (n
    /// from `noise`), then key ink with alpha `alpha[x] / 255`, to sRGB8 in `out` (3 bytes per pixel).
    pub(crate) fn row(&self, cov: [&[u8]; 3], noise: &[f32], paper_grain: f32, alpha: &[u8], out: &mut [u8]) {
        let n = out.len() / 3;
        let (c0, c1, c2) = (&cov[0][..n], &cov[1][..n], &cov[2][..n]);
        let (noise, alpha) = (&noise[..n], &alpha[..n]);
        let half = paper_grain * 0.5;
        for x in 0..n {
            let g = 1.0 + half * (2.0 * noise[x] - 1.0);
            let a = alpha[x] as f32 / 255.0;
            for ch in 0..3 {
                let mut v = self.paper[ch] * g;
                v *= self.fac[0][ch][c0[x] as usize];
                v *= self.fac[1][ch][c1[x] as usize];
                v *= self.fac[2][ch][c2[x] as usize];
                v += (v * self.trans_k[ch] - v) * a;
                out[x * 3 + ch] = self.srgb.get(v);
            }
        }
    }
}

/// Subtractive CMY mix on grained paper in linear light, then the solid key ink on top.
///
/// # Panics
/// If `ink_alpha` is given with dimensions different from the coverage planes.
pub fn composite(cov: &Coverage, ink_alpha: Option<&GrayImage>, model: &InkModel, grain: f32, seed: u64) -> RgbImage {
    let (w, h) = (cov.width, cov.height);
    let (wu, hu) = (w as usize, h as usize);
    if let Some(a) = ink_alpha {
        assert_eq!((a.width(), a.height()), (w, h), "ink alpha must match the coverage planes");
    }
    let alpha = ink_alpha.map(|a| a.as_raw().as_slice());
    let mixer = Mixer::new(model);
    let paper_seed = grain_seeds(seed)[3];
    let mut buf = vec![0u8; wu * hu * 3];
    buf.par_chunks_mut(wu * 3 * BAND).enumerate().for_each(|(b, band)| {
        let y0 = b * BAND;
        let rows = band.len() / (wu * 3);
        let fbm = (grain > 0.0).then(|| FbmBand::new(y0, rows, paper_seed, wu));
        let mut noise = vec![0f32; wu];
        let zeros = vec![0u8; wu];
        for ry in 0..rows {
            let y = y0 + ry;
            if let Some(f) = &fbm {
                f.row(y, &mut noise);
            }
            let r = y * wu..(y + 1) * wu;
            let a = alpha.map_or(&zeros[..], |a| &a[r.clone()]);
            let c = [&cov.planes[0][r.clone()], &cov.planes[1][r.clone()], &cov.planes[2][r]];
            mixer.row(c, &noise, grain, a, &mut band[ry * wu * 3..(ry + 1) * wu * 3]);
        }
    });
    RgbImage::from_raw(w, h, buf).expect("buffer size matches dimensions")
}

/// That ink alone on paper.
pub fn separation_preview(cov: &Coverage, ink: usize, model: &InkModel) -> RgbImage {
    let (w, h) = (cov.width, cov.height);
    let plane = &cov.planes[ink];
    let mut buf = vec![0u8; w as usize * h as usize * 3];
    buf.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        let mut c = [0.0; 3];
        c[ink] = plane[i] as f32 / 255.0;
        px.copy_from_slice(&lin_to_srgb8(model.mix(c, 1.0)));
    });
    RgbImage::from_raw(w, h, buf).expect("buffer size matches dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ink::PAPER_WHITE;
    use crate::types::InkSet;
    use image::Luma;

    fn flat(w: u32, h: u32, c: [u8; 3]) -> Coverage {
        let n = (w * h) as usize;
        Coverage { width: w, height: h, planes: [vec![c[0]; n], vec![c[1]; n], vec![c[2]; n]] }
    }

    fn close(a: [u8; 3], b: [u8; 3], tol: i32) -> bool {
        a.iter().zip(&b).all(|(x, y)| (*x as i32 - *y as i32).abs() <= tol)
    }

    #[test]
    fn zero_coverage_is_paper() {
        let inks = InkSet::aged();
        let img = composite(&flat(8, 8, [0; 3]), None, &InkModel::new(&inks), 0.0, 1);
        assert!(img.pixels().all(|p| close(p.0, inks.paper, 1)));
    }

    #[test]
    fn separation_preview_of_solid_ink_is_its_swatch() {
        let inks = InkSet::aged().with_paper(PAPER_WHITE);
        let m = InkModel::new(&inks);
        let sw = [inks.cyan, inks.magenta, inks.yellow];
        for k in 0..3 {
            let mut c = [0; 3];
            c[k] = 255;
            let sep = separation_preview(&flat(4, 4, c), k, &m);
            assert!(close(sep.get_pixel(0, 0).0, sw[k], 1));
            let img = composite(&flat(4, 4, c), None, &m, 0.0, 1);
            assert!(close(img.get_pixel(1, 1).0, sep.get_pixel(0, 0).0, 1));
        }
    }

    #[test]
    #[should_panic(expected = "ink alpha must match")]
    fn mismatched_ink_alpha_is_rejected() {
        let a = GrayImage::from_pixel(3, 4, Luma([255]));
        composite(&flat(4, 4, [0; 3]), Some(&a), &InkModel::new(&InkSet::aged()), 0.0, 1);
    }

    #[test]
    fn matches_per_pixel_reference() {
        let m = InkModel::new(&InkSet::aged());
        let (w, h) = (70u32, 45u32);
        let n = (w * h) as usize;
        let pl = |k: u64| (0..n).map(|i| (crate::noise::hash_u64(i as u64, k, 3) % 256) as u8).collect::<Vec<_>>();
        let cov = Coverage { width: w, height: h, planes: [pl(0), pl(1), pl(2)] };
        let a = GrayImage::from_raw(w, h, pl(3)).unwrap();
        let got = composite(&cov, Some(&a), &m, 0.08, 5);
        let ps = crate::noise::grain_seeds(5)[3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let gain = 1.0 + 0.08 * 0.5 * (2.0 * crate::noise::fbm(x as f32, y as f32, ps) - 1.0);
                let c = [0, 1, 2].map(|k| cov.planes[k][i] as f32 / 255.0);
                let want = lin_to_srgb8(m.apply_k(m.mix(c, gain), a.as_raw()[i] as f32 / 255.0));
                assert!(close(got.get_pixel(x, y).0, want, 1), "({x},{y})");
            }
        }
    }

    #[test]
    fn cmy_darker_than_any_single_ink() {
        let m = InkModel::new(&InkSet::aged());
        let lum = |p: [u8; 3]| p.iter().map(|&v| v as u32).sum::<u32>();
        let all = composite(&flat(2, 2, [255; 3]), None, &m, 0.0, 1).get_pixel(0, 0).0;
        for k in 0..3 {
            let mut c = [0; 3];
            c[k] = 255;
            let one = composite(&flat(2, 2, c), None, &m, 0.0, 1).get_pixel(0, 0).0;
            assert!(lum(all) < lum(one));
        }
    }

    #[test]
    fn full_alpha_gives_key_black() {
        let inks = InkSet::aged();
        let m = InkModel::new(&inks);
        let a = GrayImage::from_pixel(4, 4, Luma([255]));
        let img = composite(&flat(4, 4, [0; 3]), Some(&a), &m, 0.0, 1);
        assert!(close(img.get_pixel(2, 2).0, inks.black, 4), "{:?}", img.get_pixel(2, 2));
    }

    #[test]
    fn grain_varies_and_is_seeded() {
        let m = InkModel::new(&InkSet::aged());
        let c = flat(32, 32, [0; 3]);
        let a = composite(&c, None, &m, 0.1, 1);
        assert_ne!(a.as_raw(), composite(&c, None, &m, 0.1, 2).as_raw());
        assert_eq!(a.as_raw(), composite(&c, None, &m, 0.1, 1).as_raw());
        assert!(a.pixels().any(|p| p != a.get_pixel(0, 0)));
    }
}
