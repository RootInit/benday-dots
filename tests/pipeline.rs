#![cfg(feature = "cli")]

use benday_dots::Options;
use benday_dots::chart::{OUTLINE_THICKNESS, test_chart};
use benday_dots::pipeline::run_image;
use benday_dots::types::*;
#[cfg(feature = "cli")]
use std::process::Command;

fn cfg(o: Options) -> Options {
    let mut c = o;
    c.boost = false;
    c.dot_gain = 0.0;
    c.work_size = 800;
    c.dots_across = 100.0;
    c
}

#[cfg(feature = "cli")]
fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("benday_test_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn patch_hits(layout: &benday_dots::chart::ChartLayout, idx: &IndexMap, k: f32) -> usize {
    layout
        .patches
        .iter()
        .filter(|(code, (x, y, w, h))| {
            let (x0, y0) = (*x as f32 * k, *y as f32 * k);
            let (cw, ch) = (*w as f32 * k, *h as f32 * k);
            let mut ok = 0u32;
            let mut n = 0u32;
            for yy in (y0 + ch * 0.25) as u32..(y0 + ch * 0.75) as u32 {
                for xx in (x0 + cw * 0.25) as u32..(x0 + cw * 0.75) as u32 {
                    n += 1;
                    ok += (idx.get(xx, yy) == *code) as u32;
                }
            }
            ok as f32 >= 0.9 * n as f32
        })
        .count()
}

#[test]
fn chart_round_trip() {
    let c = cfg(Options::comic());
    let (img, layout) = test_chart(800, 600, &c.inks);
    let out = run_image(&img, &c).unwrap();
    assert_eq!(out.image.dimensions(), (800, 600));
    let hits = patch_hits(&layout, &out.index, 1.0);
    assert!(hits >= 60, "only {hits}/64 patches kept their code");

    let alpha = out.ink_alpha.expect("ink alpha");
    let (sx, sy, side) = layout.outline;
    let mid = OUTLINE_THICKNESS / 2;
    let mut covered = 0;
    let mut total = 0;
    for t in 0..side {
        for (x, y) in
            [(sx + t, sy + mid), (sx + t, sy + side - 1 - mid), (sx + mid, sy + t), (sx + side - 1 - mid, sy + t)]
        {
            total += 1;
            covered += (alpha.get_pixel(x, y).0[0] > 128) as u32;
        }
    }
    assert!(covered as f32 >= 0.9 * total as f32, "outline {covered}/{total}");

    let (cx, cy, r) = layout.disk;
    let mut dn = 0;
    let mut dc = 0;
    for y in 0..600u32 {
        for x in 0..800u32 {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            if dx * dx + dy * dy <= r * r {
                dn += 1;
                dc += (alpha.get_pixel(x, y).0[0] > 128) as u32;
            }
        }
    }
    assert!(dc as f32 >= 0.9 * dn as f32, "disk {dc}/{dn}");
}

#[test]
fn scale_two() {
    let c = cfg(Options::comic().scale(2.0));
    let (img, layout) = test_chart(800, 600, &c.inks);
    let out = run_image(&img, &c).unwrap();
    assert_eq!(out.image.dimensions(), (1600, 1200));
    let hits = patch_hits(&layout, &out.index, 2.0);
    assert!(hits >= 60, "only {hits}/64 patches at scale 2");
}

#[test]
fn kmeans_limits_indices() {
    let c = cfg(Options::comic().colors(Some(8)));
    let (img, _) = test_chart(800, 600, &c.inks);
    let out = run_image(&img, &c).unwrap();
    let seen: std::collections::HashSet<u8> = out.index.data.iter().copied().collect();
    assert!((2..=8).contains(&seen.len()), "{} distinct", seen.len());
}

#[test]
fn deterministic() {
    let c = cfg(Options::comic().seed(7));
    let (img, _) = test_chart(800, 600, &c.inks);
    let a = run_image(&img, &c).unwrap();
    let b = run_image(&img, &c).unwrap();
    assert_eq!(a.image.as_raw(), b.image.as_raw());
}

#[cfg(feature = "cli")]
#[test]
fn cli_smoke() {
    let d = scratch("cli");
    let chart = d.join("c.png");
    let bin = env!("CARGO_BIN_EXE_benday-dots");
    let st = Command::new(bin).arg("--test-chart").arg(&chart).args(["--chart-size", "800x600"]).status().unwrap();
    assert!(st.success() && chart.exists());
    let st = Command::new(bin).arg(&chart).arg(d.join("o.png")).arg("--dump-stages").arg(d.join("d")).status().unwrap();
    assert!(st.success());
    for f in ["01_smoothed", "02_quantized", "03_lines", "04_blacks", "05_c", "06_m", "07_y", "08_final"] {
        assert!(d.join("d").join(format!("{f}.png")).exists(), "{f} missing");
    }
    let printed = image::open(d.join("o.png")).unwrap();
    assert_eq!((printed.width(), printed.height()), (800, 600));
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn pop_style_uses_only_allowed_codes() {
    let mut c = cfg(Options::pop());
    c.dots_across = 40.0;
    assert!(c.allowed.is_some() && c.keep_dark_codes);
    let (img, _) = test_chart(800, 600, &c.inks);
    let out = run_image(&img, &c).unwrap();
    let allowed = c.allowed.clone().unwrap();
    assert!(out.index.data.iter().all(|i| allowed.contains(i)));
    let at = |x: u32, y: u32| out.index.get(x, y);
    assert_eq!(at(5, 595), 0, "white margin stays paper");
    assert!(matches!(at(10, 100), 12 | 15 | 3), "red hue band snapped to {}", at(10, 100));
    assert!(out.index.data.iter().any(|&i| i != 0 && i != 63), "no coloured regions");
}

/// Dark scene: a lit window on a dark-green field. The field must stay a coloured fill, not solid black.
#[test]
fn dark_photo_is_not_buried_in_black() {
    let mut img = image::RgbImage::new(400, 300);
    for (x, _, p) in img.enumerate_pixels_mut() {
        let t = x as f32 / 400.0;
        *p = image::Rgb([(6.0 + 34.0 * t) as u8, (16.0 + 64.0 * t) as u8, (9.0 + 40.0 * t) as u8]);
    }
    for y in 100..200 {
        for x in 150..250 {
            img.put_pixel(x, y, image::Rgb([230, 200, 120]));
        }
    }
    let mut c = cfg(Options::comic());
    c.boost = true;
    let out = run_image(&img, &c).unwrap();
    let alpha = out.ink_alpha.expect("ink alpha");
    let black = alpha.as_raw().iter().filter(|&&v| v > 128).count() as f32 / alpha.as_raw().len() as f32;
    assert!(black < 0.5, "{black:.2} of a dark-green image is black ink");
    let window = out.index.get(200, 150);
    assert_ne!(window, 63);
}

/// Neutral near-black stays solid black.
#[test]
fn neutral_black_is_solid() {
    let mut img = image::RgbImage::from_pixel(400, 300, image::Rgb([240, 235, 220]));
    for y in 60..240 {
        for x in 100..300 {
            img.put_pixel(x, y, image::Rgb([8, 8, 8]));
        }
    }
    let mut c = cfg(Options::comic());
    c.boost = true;
    let out = run_image(&img, &c).unwrap();
    assert!(out.ink_alpha.unwrap().get_pixel(200, 150)[0] > 200);
}

#[test]
fn pop_palette_has_green() {
    let green = |c: u8| {
        let c = Code::from_index(c);
        c.c > 0 && c.y > 0 && c.m == 0
    };
    assert!(POP_CODES.iter().any(|&c| green(c)));
    assert!(POP_CODES.contains(&51));
    let img = image::RgbImage::from_pixel(300, 200, image::Rgb([40, 160, 50]));
    let mut c = Options::pop();
    c.work_size = 300;
    let out = run_image(&img, &c).unwrap();
    assert!(green(out.index.get(150, 100)), "green field snapped to code {}", out.index.get(150, 100));
}
