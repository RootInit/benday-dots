use image::GrayImage;

/// Row-major 2-D buffer of `width * height` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane<T> {
    /// Columns.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Values, row by row; `data.len() == width * height`.
    pub data: Vec<T>,
}

impl<T: Copy + Send + Sync> Plane<T> {
    /// A plane filled with `fill`.
    pub fn new(width: u32, height: u32, fill: T) -> Self {
        Plane { width, height, data: vec![fill; width as usize * height as usize] }
    }

    /// Wrap existing row-major data.
    ///
    /// # Panics
    /// If `data.len() != width * height`.
    pub fn from_vec(width: u32, height: u32, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width as usize * height as usize, "Plane data length mismatch");
        Plane { width, height, data }
    }

    /// The value at column `x`, row `y`; panics when out of bounds.
    pub fn get(&self, x: u32, y: u32) -> T {
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Set the value at column `x`, row `y`; panics when out of bounds.
    pub fn set(&mut self, x: u32, y: u32, v: T) {
        self.data[y as usize * self.width as usize + x as usize] = v;
    }
}

/// Clamped bilinear footprint: (x0, y0, x1, y1, fx, fy).
fn footprint(w: u32, h: u32, x: f32, y: f32) -> (usize, usize, usize, usize, f32, f32) {
    assert!(w > 0 && h > 0, "cannot sample an empty plane");
    let xc = x.clamp(0.0, (w - 1) as f32);
    let yc = y.clamp(0.0, (h - 1) as f32);
    let x0 = xc.floor();
    let y0 = yc.floor();
    let x1 = (x0 as usize + 1).min(w as usize - 1);
    let y1 = (y0 as usize + 1).min(h as usize - 1);
    (x0 as usize, y0 as usize, x1, y1, xc - x0, yc - y0)
}

impl Plane<f32> {
    /// Clamped bilinear sample at pixel-index coordinates; panics on an empty plane.
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let (x0, y0, x1, y1, fx, fy) = footprint(self.width, self.height, x, y);
        let w = self.width as usize;
        let top = self.data[y0 * w + x0] * (1.0 - fx) + self.data[y0 * w + x1] * fx;
        let bot = self.data[y1 * w + x0] * (1.0 - fx) + self.data[y1 * w + x1] * fx;
        top * (1.0 - fy) + bot * fy
    }
}

impl Plane<[f32; 3]> {
    /// Clamped bilinear sample at pixel-index coordinates; panics on an empty plane.
    pub fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let (x0, y0, x1, y1, fx, fy) = footprint(self.width, self.height, x, y);
        let w = self.width as usize;
        let mut out = [0.0; 3];
        for (k, o) in out.iter_mut().enumerate() {
            let top = self.data[y0 * w + x0][k] * (1.0 - fx) + self.data[y0 * w + x1][k] * fx;
            let bot = self.data[y1 * w + x0][k] * (1.0 - fx) + self.data[y1 * w + x1][k] * fx;
            *o = top * (1.0 - fy) + bot * fy;
        }
        out
    }
}

/// Lab triples on a grid.
pub type LabPlane = Plane<[f32; 3]>;
/// One float per pixel.
pub type FloatPlane = Plane<f32>;
/// One YRB code (0..64) per pixel.
pub type IndexMap = Plane<u8>;
/// 0 or 255, 255 = ink.
pub type Mask = GrayImage;

pub use crate::codes::{LEVELS, POP_CODES};

/// One YRB code: level index 0..=3 per ink. Y = yellow, R = magenta, B = cyan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Code {
    /// Cyan (B) level, 0 to 3.
    pub c: u8,
    /// Magenta (R) level, 0 to 3.
    pub m: u8,
    /// Yellow (Y) level, 0 to 3.
    pub y: u8,
}

impl Code {
    /// The code number `c*16 + m*4 + y`; every level must be at most 3.
    pub fn index(self) -> u8 {
        debug_assert!(self.c < 4 && self.m < 4 && self.y < 4, "levels must be 0..=3: {self:?}");
        self.c * 16 + self.m * 4 + self.y
    }

    /// Inverse of [`Code::index`]; panics when `i >= 64`.
    pub fn from_index(i: u8) -> Code {
        assert!(i < 64, "code index {i} out of range");
        Code { c: i / 16, m: (i / 4) % 4, y: i % 4 }
    }

    /// [c, m, y] in 0..1.
    pub fn tints(self, levels: &[f32; 4]) -> [f32; 3] {
        [levels[self.c as usize], levels[self.m as usize], levels[self.y as usize]]
    }
}

/// Per-pixel tints handed to the screen: a code per pixel, looked up in `levels`. Ink order everywhere:
/// 0 = C, 1 = M, 2 = Y.
pub struct TintField {
    /// Code of each pixel.
    pub index: IndexMap,
    /// `[c, m, y]` tints per code.
    pub levels: Vec<[f32; 3]>,
}

impl TintField {
    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.index.width
    }

    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.index.height
    }

    /// Tint of ink `ink` at pixel (x, y), 0..=1.
    pub fn tint(&self, x: u32, y: u32, ink: usize) -> f32 {
        self.levels[self.index.get(x, y) as usize][ink]
    }
}

/// Screened area coverage per ink, 0..=255 (= 0..1), output grid. `planes[ink].len()` == w*h.
pub struct Coverage {
    /// Columns.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Coverage of C, M and Y.
    pub planes: [Vec<u8>; 3],
}

/// Ink and paper colours as sRGB8 swatches ("ink on white").
#[derive(Clone, Copy, Debug)]
pub struct InkSet {
    /// Cyan.
    pub cyan: [u8; 3],
    /// Magenta.
    pub magenta: [u8; 3],
    /// Yellow.
    pub yellow: [u8; 3],
    /// Black (key) ink.
    pub black: [u8; 3],
    /// Paper.
    pub paper: [u8; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_round_trip() {
        for i in 0..64u8 {
            assert_eq!(Code::from_index(i).index(), i);
        }
        let c = Code::from_index(0);
        assert_eq!((c.c, c.m, c.y), (0, 0, 0));
    }

    #[test]
    fn code_tints() {
        assert_eq!(Code { c: 3, m: 2, y: 1 }.tints(&LEVELS), [1.0, 0.5, 0.25]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn code_from_index_panics() {
        Code::from_index(64);
    }

    #[test]
    fn bilinear_f32() {
        let p = Plane::from_vec(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(p.bilinear(x as f32, y as f32), p.get(x, y));
            }
        }
        assert!((p.bilinear(0.5, 0.0) - 0.5).abs() < 1e-6);
        assert!((p.bilinear(0.5, 0.5) - 1.5).abs() < 1e-6);
        assert_eq!(p.bilinear(-5.0, -5.0), 0.0);
        assert_eq!(p.bilinear(9.0, 9.0), 3.0);
    }

    #[test]
    fn bilinear_rgb() {
        let p = Plane::from_vec(2, 1, vec![[0.0, 10.0, 0.0], [1.0, 20.0, 0.0]]);
        assert_eq!(p.bilinear(1.0, 0.0), [1.0, 20.0, 0.0]);
        let m = p.bilinear(0.5, 0.0);
        assert!((m[0] - 0.5).abs() < 1e-6 && (m[1] - 15.0).abs() < 1e-5);
        assert_eq!(p.bilinear(-3.0, 4.0), [0.0, 10.0, 0.0]);
    }

    #[test]
    fn tint_field_lookup() {
        let mut levels = vec![[0.0; 3]; 64];
        levels[5] = [0.25, 0.5, 1.0];
        let f = TintField { index: Plane::new(2, 2, 5u8), levels };
        assert_eq!((f.width(), f.height()), (2, 2));
        assert_eq!(f.tint(1, 1, 1), 0.5);
    }
}
