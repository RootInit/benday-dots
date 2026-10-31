use benday_dots::codes::RenderedPalette;
use benday_dots::composite::composite;
use benday_dots::ink::{InkModel, delta_e, lin_to_lab, srgb8_to_lin};
use benday_dots::screen::{BENDAY_ANGLES, ScreenConfig, ScreenParams, screen};
use benday_dots::types::*;

fn seam(pitch: f32) {
    let inks = InkSet::aged();
    let model = InkModel::new(&inks);
    let pal = RenderedPalette::build(&model, &LEVELS, pitch, 0.25);
    let side = (16.0 * pitch).ceil().max(256.0) as u32;
    let cfg = ScreenConfig::from_params(ScreenParams {
        pitch,
        dot_gain: 0.25,
        grain: 0.0,
        misregister: 0.0,
        angles: BENDAY_ANGLES,
        seed: 1,
    });
    let mut worst = 0.0f32;
    for i in 0..64u8 {
        let index = IndexMap::new(side, side, i);
        let field = TintField { index, levels: pal.tints.clone() };
        let cov = screen(&field, &cfg);
        let img = composite(&cov, None, &model, 0.0, 1);
        let mut sum = [0.0f64; 3];
        for p in img.pixels() {
            let l = srgb8_to_lin(p.0);
            for k in 0..3 {
                sum[k] += l[k] as f64;
            }
        }
        let n = (side * side) as f64;
        let mean = [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32];
        let de = delta_e(lin_to_lab(mean), pal.lab[i as usize], 1.5);
        worst = worst.max(de);
        assert!(de < 2.0, "pitch {pitch} code {i}: dE {de}");
    }
    eprintln!("pitch {pitch}: worst dE {worst}");
}

#[test]
fn seam_pitch_10() {
    seam(10.0);
}

#[test]
fn seam_pitch_27() {
    seam(27.0);
}
