//! 8-bit RGB PNG writer that filters and deflates horizontal bands in parallel.
//!
//! Each band becomes one IDAT chunk; the zlib stream is their concatenation (see [`crate::deflate`]).

use crate::deflate::{adler32, adler32_combine, compress_band, stored_band};
use crate::io::PngCompression;
use image::RgbImage;
use rayon::prelude::*;
use std::io::{self, Write};

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
const BPP: usize = 3;
/// Rows per band: enough bands to keep every core busy on a photo, few enough that the per-band block header
/// (about 54 bytes) is noise.
const BAND_ROWS: usize = 64;
/// Upper bound on a band's raw size, so that absurdly wide images still give IDAT chunks well under 2 GiB.
const BAND_BYTES: usize = 16 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    None = 0,
    Sub = 1,
    Up = 2,
    Avg = 3,
    Paeth = 4,
}

/// Paeth predictor on unsigned bytes only, which vectorises (after FPNGE, as used by the `png` crate).
#[inline(always)]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let pa = b.max(c) - b.min(c);
    let pb = a.max(c) - a.min(c);
    let pc = if (a < c) == (c < b) { pa.max(pb) - pa.min(pb) } else { 255 };
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn filter_row(f: Filter, cur: &[u8], prev: &[u8], out: &mut [u8]) {
    let n = BPP.min(cur.len());
    let (left, above_left) = (&cur[..cur.len() - n], &prev[..cur.len() - n]);
    match f {
        Filter::None => out.copy_from_slice(cur),
        Filter::Up => {
            for ((o, &c), &p) in out.iter_mut().zip(cur).zip(prev) {
                *o = c.wrapping_sub(p);
            }
        }
        Filter::Sub => {
            out[..n].copy_from_slice(&cur[..n]);
            for ((o, &c), &a) in out[n..].iter_mut().zip(&cur[n..]).zip(left) {
                *o = c.wrapping_sub(a);
            }
        }
        Filter::Avg => {
            for ((o, &c), &b) in out[..n].iter_mut().zip(cur).zip(prev) {
                *o = c.wrapping_sub(b / 2);
            }
            for (((o, &c), &a), &b) in out[n..].iter_mut().zip(&cur[n..]).zip(left).zip(&prev[n..]) {
                *o = c.wrapping_sub((a & b) + ((a ^ b) >> 1));
            }
        }
        Filter::Paeth => {
            for ((o, &c), &b) in out[..n].iter_mut().zip(cur).zip(prev) {
                *o = c.wrapping_sub(b);
            }
            for ((((o, &c), &a), &b), &d) in
                out[n..].iter_mut().zip(&cur[n..]).zip(left).zip(&prev[n..]).zip(above_left)
            {
                *o = c.wrapping_sub(paeth(a, b, d));
            }
        }
    }
}

/// Heuristic cost of a filtered row: the sum of the bytes read as signed magnitudes.
fn cost(row: &[u8]) -> u64 {
    let (chunks, rest) = row.as_chunks::<32>();
    let mut sum = 0u64;
    for c in chunks {
        // At most 32 * 128, so a u32 accumulator keeps the inner loop vectorisable.
        sum += u64::from(c.iter().map(|&b| u32::from((b as i8).unsigned_abs())).sum::<u32>());
    }
    sum + rest.iter().map(|&b| u64::from((b as i8).unsigned_abs())).sum::<u64>()
}

/// Filter `rows` of the image starting at row `y0` into `out` (filter byte, then the row, per row).
fn filter_band(raw: &[u8], stride: usize, y0: usize, rows: usize, mode: PngCompression, out: &mut Vec<u8>) {
    let zeros = vec![0u8; stride];
    let mut best = vec![0u8; stride];
    let mut cand = vec![0u8; stride];
    for y in y0..y0 + rows {
        let cur = &raw[y * stride..(y + 1) * stride];
        let prev = if y == 0 { &zeros[..] } else { &raw[(y - 1) * stride..y * stride] };
        match mode {
            PngCompression::None => {
                out.push(Filter::None as u8);
                out.extend_from_slice(cur);
            }
            PngCompression::Fastest => {
                out.push(Filter::Up as u8);
                let at = out.len();
                out.resize(at + stride, 0);
                filter_row(Filter::Up, cur, prev, &mut out[at..]);
            }
            PngCompression::Fast => {
                let (mut pick, mut min) = (Filter::Up, u64::MAX);
                for f in [Filter::Up, Filter::Sub, Filter::Avg, Filter::Paeth] {
                    filter_row(f, cur, prev, &mut cand);
                    let c = cost(&cand);
                    if c <= min {
                        (pick, min) = (f, c);
                        std::mem::swap(&mut best, &mut cand);
                        if c == 0 {
                            break;
                        }
                    }
                }
                out.push(pick as u8);
                out.extend_from_slice(&best);
            }
        }
    }
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() + 12);
    v.extend_from_slice(&(data.len() as u32).to_be_bytes());
    v.extend_from_slice(kind);
    v.extend_from_slice(data);
    finish_chunk(&mut v);
    v
}

/// Append the CRC to a chunk laid out as length, type, data.
fn finish_chunk(v: &mut Vec<u8>) {
    let crc = crc32fast::hash(&v[4..]);
    v.extend_from_slice(&crc.to_be_bytes());
}

/// Write `img` as an 8-bit RGB PNG.
pub(crate) fn write_rgb8<W: Write>(mut w: W, img: &RgbImage, mode: PngCompression) -> io::Result<()> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    if width == 0 || height == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "PNG images cannot be empty"));
    }
    let stride = width * BPP;
    let raw = img.as_raw();
    let rows_per_band = (BAND_BYTES / (stride + 1).max(1)).clamp(1, BAND_ROWS);
    let bands = height.div_ceil(rows_per_band);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&img.width().to_be_bytes());
    ihdr.extend_from_slice(&img.height().to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    // (chunk without its CRC, Adler-32 of the band's raw data, raw length)
    let mut parts: Vec<(Vec<u8>, u32, u64)> = (0..bands)
        .into_par_iter()
        .map(|b| {
            let y0 = b * rows_per_band;
            let rows = rows_per_band.min(height - y0);
            let mut filtered = Vec::with_capacity(rows * (stride + 1));
            filter_band(raw, stride, y0, rows, mode, &mut filtered);
            let (first, last) = (b == 0, b + 1 == bands);
            let body = if mode == PngCompression::None {
                stored_band(&filtered, first, last)
            } else {
                compress_band(&filtered, first, last)
            };
            let mut c = Vec::with_capacity(body.len() + 16);
            c.extend_from_slice(&[0; 4]);
            c.extend_from_slice(b"IDAT");
            c.extend_from_slice(&body);
            (c, adler32(&filtered), filtered.len() as u64)
        })
        .collect();

    let adler = parts.iter().skip(1).fold(parts[0].1, |acc, p| adler32_combine(acc, p.1, p.2));
    if let Some((last, ..)) = parts.last_mut() {
        last.extend_from_slice(&adler.to_be_bytes());
    }
    parts.par_iter_mut().for_each(|(c, ..)| {
        let len = (c.len() - 8) as u32;
        c[..4].copy_from_slice(&len.to_be_bytes());
        finish_chunk(c);
    });

    w.write_all(&SIGNATURE)?;
    w.write_all(&chunk(b"IHDR", &ihdr))?;
    for (c, ..) in &parts {
        w.write_all(c)?;
    }
    w.write_all(&chunk(b"IEND", &[]))?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inverse of [`filter_row`] following the PNG specification literally.
    fn unfilter(f: Filter, filtered: &[u8], prev: &[u8]) -> Vec<u8> {
        let mut cur = vec![0u8; filtered.len()];
        for i in 0..filtered.len() {
            let a = if i >= BPP { i32::from(cur[i - BPP]) } else { 0 };
            let b = i32::from(prev[i]);
            let c = if i >= BPP { i32::from(prev[i - BPP]) } else { 0 };
            let pred = match f {
                Filter::None => 0,
                Filter::Sub => a,
                Filter::Up => b,
                Filter::Avg => (a + b) / 2,
                Filter::Paeth => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
            };
            cur[i] = filtered[i].wrapping_add(pred as u8);
        }
        cur
    }

    #[test]
    fn every_filter_inverts_per_the_spec() {
        let mut s = 7u32;
        let mut rnd = move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        };
        for len in [3usize, 6, 33, 96, 300] {
            let cur: Vec<u8> = (0..len).map(|_| rnd()).collect();
            let prev: Vec<u8> = (0..len).map(|_| rnd()).collect();
            for f in [Filter::None, Filter::Sub, Filter::Up, Filter::Avg, Filter::Paeth] {
                let mut out = vec![0u8; len];
                filter_row(f, &cur, &prev, &mut out);
                assert_eq!(unfilter(f, &out, &prev), cur, "{} len {len}", f as u8);
            }
        }
    }
}
