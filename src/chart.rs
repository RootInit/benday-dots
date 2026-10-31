use crate::codes::RenderedPalette;
use crate::ink::{InkModel, lab_to_srgb8};
use crate::types::{InkSet, LEVELS};
use image::{Rgb, RgbImage};

/// Where the chart's features are, in chart pixels.
pub struct ChartLayout {
    /// Code index, rect (x, y, w, h).
    pub patches: Vec<(u8, (u32, u32, u32, u32))>,
    /// The black-outlined square, as (x, y, side); the outline is [`OUTLINE_THICKNESS`] px thick.
    pub outline: (u32, u32, u32),
    /// The near-black disk, as (centre x, centre y, radius).
    pub disk: (f32, f32, f32),
}

/// Thickness of the black square outline in px.
pub const OUTLINE_THICKNESS: u32 = 6;

fn hsv(h: f32, s: f32, v: f32) -> [u8; 3] {
    let k = |n: f32| {
        let kk = (n + h * 6.0) % 6.0;
        v - v * s * kk.min(4.0 - kk).clamp(0.0, 1.0)
    };
    [k(5.0), k(3.0), k(1.0)].map(|c| (c * 255.0).round() as u8)
}

/// A synthetic test card: grey and hue ramps, the 64 code patches on mid grey, an outlined square and a dark disk.
pub fn test_chart(w: u32, h: u32, inks: &InkSet) -> (RgbImage, ChartLayout) {
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
    let band = (h as f32 * 0.12).round() as u32;
    for x in 0..w {
        let g = (x as f32 / (w - 1).max(1) as f32 * 255.0).round() as u8;
        let hue = hsv(x as f32 / w as f32, 1.0, 1.0);
        for y in 0..band {
            img.put_pixel(x, y, Rgb([g; 3]));
            img.put_pixel(x, band + y, Rgb(hue));
        }
    }

    let model = InkModel::new(inks);
    let pal = RenderedPalette::build(&model, &LEVELS, 10.0, 0.0);
    let gy = 2 * band;
    let gh = (h as f32 * 0.5).round() as u32;
    let cell_w = w as f32 / 8.0;
    let cell_h = gh as f32 / 8.0;
    let (pw, ph) = ((cell_w * 0.96).floor() as u32, (cell_h * 0.96).floor() as u32);
    for y in gy..(gy + gh).min(h) {
        for x in 0..w {
            img.put_pixel(x, y, Rgb([128, 128, 128]));
        }
    }
    let mut patches = Vec::with_capacity(64);
    for i in 0..64u8 {
        let (col, row) = ((i % 8) as f32, (i / 8) as f32);
        let x0 = (col * cell_w + (cell_w - pw as f32) / 2.0).round() as u32;
        let y0 = gy + (row * cell_h + (cell_h - ph as f32) / 2.0).round() as u32;
        for y in y0..(y0 + ph).min(h) {
            for x in x0..(x0 + pw).min(w) {
                img.put_pixel(x, y, Rgb(pal.srgb[i as usize]));
            }
        }
        patches.push((i, (x0, y0, pw, ph)));
    }

    let ry = gy + gh;
    let rh = h.saturating_sub(ry);
    let grey = Rgb(lab_to_srgb8([60.0, 0.0, 0.0]));
    let side = (rh as f32 * 0.7).round() as u32;
    let (sx, sy) = (w / 10, ry + (rh - side) / 2);
    for y in sy..sy + side {
        for x in sx..sx + side {
            let t = OUTLINE_THICKNESS;
            let edge =
                x < sx + t || x >= (sx + side).saturating_sub(t) || y < sy + t || y >= (sy + side).saturating_sub(t);
            img.put_pixel(x, y, if edge { Rgb([0, 0, 0]) } else { grey });
        }
    }
    let r = 0.08 * h as f32;
    let (cx, cy) = (w as f32 * 0.7, ry as f32 + rh as f32 / 2.0);
    let dark = Rgb(lab_to_srgb8([5.0, 0.0, 0.0]));
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            if dx * dx + dy * dy <= r * r {
                img.put_pixel(x, y, dark);
            }
        }
    }
    (img, ChartLayout { patches, outline: (sx, sy, side), disk: (cx, cy, r) })
}
