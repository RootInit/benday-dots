/// SplitMix64-style finaliser over a combination of the inputs.
pub fn hash_u64(a: u64, b: u64, seed: u64) -> u64 {
    let mut z = a
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F).rotate_left(23))
        .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9).rotate_left(41))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0,1).
pub fn hash01(x: i64, y: i64, seed: u64) -> f32 {
    (hash_u64(x as u64, y as u64, seed) >> 40) as f32 * (1.0 / 16_777_216.0)
}

/// Salt for the per-ink tint grain seed.
pub const SALT_GRAIN: u64 = 7;
/// Lattice index (the `y` argument of `hash01`) of the per-ink screen phase along u.
pub const SALT_PHASE_U: i64 = 0;
/// Lattice index of the per-ink screen phase along v.
pub const SALT_PHASE_V: i64 = 1;
/// Lattice index of the per-ink misregistration direction.
pub const SALT_MISREG: i64 = 2;
/// XORed into the user seed for the paper-grain noise.
pub const SALT_PAPER: u64 = 0xBEEF;

/// Noise seeds for the C, M, Y tint grain and the paper grain.
pub fn grain_seeds(seed: u64) -> [u64; 4] {
    [hash_u64(seed, 0, SALT_GRAIN), hash_u64(seed, 1, SALT_GRAIN), hash_u64(seed, 2, SALT_GRAIN), seed ^ SALT_PAPER]
}

/// Cubic smoothstep on 0..=1.
#[inline(always)]
pub fn smooth01(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Bilinear-smoothstep of a hash01 lattice, [0,1).
pub fn value_noise(x: f32, y: f32, cell: f32, seed: u64) -> f32 {
    let gx = x / cell;
    let gy = y / cell;
    let fx0 = gx.floor();
    let fy0 = gy.floor();
    let (ix, iy) = (fx0 as i64, fy0 as i64);
    let (tx, ty) = (smooth01(gx - fx0), smooth01(gy - fy0));
    let a = hash01(ix, iy, seed);
    let b = hash01(ix + 1, iy, seed);
    let c = hash01(ix, iy + 1, seed);
    let d = hash01(ix + 1, iy + 1, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    // Convex combination of values < 1 can round up to exactly 1.0 in f32.
    (top + (bot - top) * ty).min(1.0 - f32::EPSILON)
}

/// 3 octaves, cells 8, 4, 2 px, weights .5 .3 .2.
pub fn fbm(x: f32, y: f32, seed: u64) -> f32 {
    let v = 0.5 * value_noise(x, y, 8.0, seed)
        + 0.3 * value_noise(x, y, 4.0, seed.wrapping_add(1))
        + 0.2 * value_noise(x, y, 2.0, seed.wrapping_add(2));
    v.min(1.0 - f32::EPSILON)
}

const LIM: f32 = 1.0 - f32::EPSILON;

/// fbm for a band of rows. Each lattice row is hashed and x-interpolated once, then shared by the `cell`
/// output rows it spans.
pub struct FbmBand {
    oct: [(u32, usize, Vec<f32>); 3],
    w: usize,
}

impl FbmBand {
    /// Prepare rows `y0..y0 + rows` for x in `0..width`.
    pub fn new(y0: usize, rows: usize, seed: u64, width: usize) -> Self {
        assert!(rows > 0);
        let oct = [(3u32, seed), (2, seed.wrapping_add(1)), (1, seed.wrapping_add(2))].map(|(sh, s)| {
            let c = 1usize << sh;
            let iy0 = y0 >> sh;
            let iy1 = ((y0 + rows - 1) >> sh) + 1;
            let txs: Vec<f32> = (0..c).map(|r| smooth01(r as f32 / c as f32)).collect();
            let mut hrows = vec![0f32; (iy1 - iy0 + 1) * width];
            let mut lat = vec![0f32; (width >> sh) + 2];
            for (j, hr) in hrows.chunks_mut(width).enumerate() {
                for (ix, l) in lat.iter_mut().enumerate() {
                    *l = hash01(ix as i64, (iy0 + j) as i64, s);
                }
                for (x, o) in hr.iter_mut().enumerate() {
                    let (a, b) = (lat[x >> sh], lat[(x >> sh) + 1]);
                    *o = a + (b - a) * txs[x & (c - 1)];
                }
            }
            (sh, iy0, hrows)
        });
        FbmBand { oct, w: width }
    }

    /// `out[x] == fbm(x, y, seed)` bit for bit, for `y` inside the band and `x < min(width, out.len())`.
    pub fn row(&self, y: usize, out: &mut [f32]) {
        let w = self.w.min(out.len());
        let get = |o: usize| {
            let (sh, iy0, ref h) = self.oct[o];
            let c = 1usize << sh;
            let j = (y >> sh) - iy0;
            let ty = smooth01((y & (c - 1)) as f32 / c as f32);
            (&h[j * self.w..j * self.w + w], &h[(j + 1) * self.w..(j + 1) * self.w + w], ty)
        };
        let ((t0, b0, y0), (t1, b1, y1), (t2, b2, y2)) = (get(0), get(1), get(2));
        let out = &mut out[..w];
        for x in 0..w {
            let v0 = (t0[x] + (b0[x] - t0[x]) * y0).min(LIM);
            let v1 = (t1[x] + (b1[x] - t1[x]) * y1).min(LIM);
            let v2 = (t2[x] + (b2[x] - t2[x]) * y2).min(LIM);
            out[x] = (0.5 * v0 + 0.3 * v1 + 0.2 * v2).min(LIM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_in_range() {
        for i in 0..1000 {
            let h = hash01(i, -i * 7, 3);
            assert!((0.0..1.0).contains(&h));
            assert_eq!(h, hash01(i, -i * 7, 3));
        }
        assert_ne!(hash_u64(1, 2, 3), hash_u64(2, 1, 3));
        assert_ne!(hash01(5, 5, 1), hash01(5, 5, 2));
    }

    #[test]
    fn hash01_mean_near_half() {
        let m: f32 = (0..10_000).map(|i| hash01(i, i / 100, 9)).sum::<f32>() / 10_000.0;
        assert!((m - 0.5).abs() < 0.02, "{m}");
    }

    #[test]
    fn value_noise_matches_lattice() {
        assert!((value_noise(16.0, 24.0, 8.0, 4) - hash01(2, 3, 4)).abs() < 1e-6);
    }

    #[test]
    fn fbm_range_determinism_mean() {
        let mut sum = 0.0f64;
        for y in 0..256 {
            for x in 0..256 {
                let v = fbm(x as f32, y as f32, 11);
                assert!((0.0..1.0).contains(&v));
                assert_eq!(v, fbm(x as f32, y as f32, 11));
                sum += v as f64;
            }
        }
        let mean = sum / 65536.0;
        assert!((0.4..=0.6).contains(&mean), "{mean}");
    }

    #[test]
    fn fbm_band_is_bit_identical() {
        let w = 301;
        let mut row = vec![0f32; w];
        for (y0, rows) in [(0usize, 32usize), (37, 20), (3000, 24)] {
            let b = FbmBand::new(y0, rows, 99, w);
            for y in y0..y0 + rows {
                b.row(y, &mut row);
                for (x, &v) in row.iter().enumerate() {
                    assert_eq!(v.to_bits(), fbm(x as f32, y as f32, 99).to_bits(), "{x},{y}");
                }
            }
        }
    }
}
