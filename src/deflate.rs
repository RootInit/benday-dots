//! Fast deflate for PNG data, adapted from fdeflate 0.3.7, (c) The image-rs Developers, MIT or Apache-2.0.
//! Changes: output goes to a `Vec`, the Adler-32 is left to the caller, and non-final blocks end on a byte
//! boundary so independently compressed bands can be concatenated.

use simd_adler32::Adler32;

/// Hard-coded Huffman code lengths, one fixed tree for every block; they suit filtered PNG rows.
const HUFFMAN_LENGTHS: [u8; 286] = [
    2, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9,
    9, 9, 9, 8, 9, 8, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 4, 3, 12, 12, 12, 9, 9, 11, 10, 11, 11, 10, 11, 11, 11, 11, 11,
    11, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 9,
];

/// Canonical Huffman codes (bit-reversed for LSB-first output) for `lengths`.
const fn compute_codes<const N: usize>(lengths: &[u8; N]) -> [u16; N] {
    let mut codes = [0u16; N];
    let mut code = 0u32;
    let mut len = 1;
    while len <= 16 {
        let mut i = 0;
        while i < N {
            if lengths[i] == len {
                codes[i] = (code as u16).reverse_bits() >> (16 - len);
                code += 1;
            }
            i += 1;
        }
        code <<= 1;
        len += 1;
    }
    assert!(code == 2 << 16, "HUFFMAN_LENGTHS is not a complete code");
    codes
}

const HUFFMAN_CODES: [u16; 286] = compute_codes(&HUFFMAN_LENGTHS);

/// Length code for length values (derived from the deflate spec).
const LENGTH_TO_SYMBOL: [u16; 256] = [
    257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266, 266, 267, 267, 268, 268, 269, 269, 269, 269, 270, 270, 270,
    270, 271, 271, 271, 271, 272, 272, 272, 272, 273, 273, 273, 273, 273, 273, 273, 273, 274, 274, 274, 274, 274, 274,
    274, 274, 275, 275, 275, 275, 275, 275, 275, 275, 276, 276, 276, 276, 276, 276, 276, 276, 277, 277, 277, 277, 277,
    277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278,
    278, 278, 278, 278, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 280, 280, 280,
    280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
    281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 282,
    282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282,
    282, 282, 282, 282, 282, 282, 282, 282, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
    283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 284, 284, 284, 284, 284, 284,
    284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
    284, 284, 285,
];

/// Number of extra bits for length values.
const LENGTH_TO_LEN_EXTRA: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0,
];
const BITMASKS: [u32; 17] = [
    0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF,
    0x3FFF, 0x7FFF, 0xFFFF,
];

/// The 2-byte zlib header and the dynamic-block header that defines [`HUFFMAN_LENGTHS`], ending in 5 bits.
const HEADER: [u8; 54] = [
    120, 1, 237, 192, 3, 160, 36, 89, 150, 198, 241, 255, 119, 238, 141, 200, 204, 167, 114, 75, 99, 174, 109, 219,
    182, 109, 219, 182, 109, 219, 182, 109, 105, 140, 158, 150, 74, 175, 158, 50, 51, 34, 238, 249, 118, 183, 106, 122,
    166, 135, 59, 107, 213, 15,
];
/// Bit 0 of this byte is the block's BFINAL flag.
const BFINAL_BYTE: usize = 2;

struct Bits {
    buffer: u64,
    nbits: u8,
    out: Vec<u8>,
}

impl Bits {
    fn write(&mut self, bits: u64, nbits: u8) {
        debug_assert!(nbits <= 64);
        self.buffer |= bits << self.nbits;
        self.nbits += nbits;
        if self.nbits >= 64 {
            self.out.extend_from_slice(&self.buffer.to_le_bytes());
            self.nbits -= 64;
            self.buffer = bits.checked_shr((nbits - self.nbits) as u32).unwrap_or(0);
        }
        debug_assert!(self.nbits < 64);
    }

    fn symbol(&mut self, sym: u8) {
        self.write(u64::from(HUFFMAN_CODES[sym as usize]), HUFFMAN_LENGTHS[sym as usize]);
    }

    /// Pad to a byte boundary and flush the bit buffer.
    fn align(&mut self) {
        if !self.nbits.is_multiple_of(8) {
            self.write(0, 8 - self.nbits % 8);
        }
        if self.nbits > 0 {
            self.out.extend_from_slice(&self.buffer.to_le_bytes()[..self.nbits as usize / 8]);
            self.buffer = 0;
            self.nbits = 0;
        }
    }

    fn run(&mut self, mut run: u32) {
        self.write(u64::from(HUFFMAN_CODES[0]), HUFFMAN_LENGTHS[0]);
        run -= 1;
        while run >= 258 {
            self.write(u64::from(HUFFMAN_CODES[285]), HUFFMAN_LENGTHS[285] + 1);
            run -= 258;
        }
        if run > 4 {
            let sym = LENGTH_TO_SYMBOL[run as usize - 3] as usize;
            self.write(u64::from(HUFFMAN_CODES[sym]), HUFFMAN_LENGTHS[sym]);
            let extra_bits = LENGTH_TO_LEN_EXTRA[run as usize - 3];
            let extra = u64::from((run - 3) & BITMASKS[extra_bits as usize]);
            self.write(extra, extra_bits + 1);
        } else {
            debug_assert_eq!(HUFFMAN_CODES[0], 0);
            self.write(0, run as u8 * HUFFMAN_LENGTHS[0]);
        }
    }

    fn data(&mut self, data: &[u8]) {
        let mut run = 0;
        let (chunks, rest) = data.as_chunks::<8>();
        for chunk in chunks {
            let word = u64::from_le_bytes(*chunk);
            if word == 0 {
                run += 8;
                continue;
            } else if run > 0 {
                let run_extra = word.trailing_zeros() / 8;
                self.run(run + run_extra);
                run = 0;
                if run_extra > 0 {
                    run = word.leading_zeros() / 8;
                    for &b in &chunk[run_extra as usize..8 - run as usize] {
                        self.symbol(b);
                    }
                    continue;
                }
            }
            let run_start = word.leading_zeros() / 8;
            if run_start > 0 {
                for &b in &chunk[..8 - run_start as usize] {
                    self.symbol(b);
                }
                run = run_start;
                continue;
            }
            for half in chunk.as_chunks::<4>().0 {
                let n: [u8; 4] = std::array::from_fn(|i| HUFFMAN_LENGTHS[half[i] as usize]);
                let c = |i: usize| u64::from(HUFFMAN_CODES[half[i] as usize]);
                let bits = c(0) | (c(1) << n[0]) | (c(2) << (n[0] + n[1])) | (c(3) << (n[0] + n[1] + n[2]));
                self.write(bits, n[0] + n[1] + n[2] + n[3]);
            }
        }
        if run > 0 {
            self.run(run);
        }
        for &b in rest {
            self.symbol(b);
        }
    }
}

/// Compress `data` as one deflate block. The zlib header is written when `first`. A `last` block ends the
/// stream (the caller appends the big-endian Adler-32 of all the data); any other block is followed by an empty
/// stored block so it ends on a byte boundary and the next block's output can be appended directly.
#[must_use]
pub(crate) fn compress_band(data: &[u8], first: bool, last: bool) -> Vec<u8> {
    let mut bits = Bits { buffer: 0, nbits: 0, out: Vec::with_capacity(data.len() / 4 + 128) };
    if first {
        bits.out.extend_from_slice(&HEADER[..2]);
    }
    bits.out.push(if last { HEADER[BFINAL_BYTE] } else { HEADER[BFINAL_BYTE] & !1 });
    bits.out.extend_from_slice(&HEADER[BFINAL_BYTE + 1..53]);
    bits.write(u64::from(HEADER[53]), 5);
    bits.data(data);
    bits.write(u64::from(HUFFMAN_CODES[256]), HUFFMAN_LENGTHS[256]);
    if !last {
        bits.write(0, 3);
    }
    bits.align();
    if !last {
        bits.out.extend_from_slice(&[0, 0, 0xFF, 0xFF]);
    }
    bits.out
}

/// Wrap `data` in stored blocks. Zlib header when `first`, final block flag when `last`.
#[must_use]
pub(crate) fn stored_band(data: &[u8], first: bool, last: bool) -> Vec<u8> {
    const MAX: usize = u16::MAX as usize;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX * 5 + 7);
    if first {
        out.extend_from_slice(&[0x78, 0x01]);
    }
    let mut blocks = data.chunks(MAX).peekable();
    while let Some(block) = blocks.next() {
        let len = block.len() as u16;
        out.push(u8::from(last && blocks.peek().is_none()));
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out
}

/// Adler-32 of `data`.
#[must_use]
pub(crate) fn adler32(data: &[u8]) -> u32 {
    let mut a = Adler32::new();
    a.write(data);
    a.finish()
}

/// Adler-32 of the concatenation of two inputs, from their checksums and the length of the second.
#[must_use]
pub(crate) fn adler32_combine(a1: u32, a2: u32, len2: u64) -> u32 {
    const BASE: u32 = 65521;
    let rem = (len2 % u64::from(BASE)) as u32;
    let mut sum1 = a1 & 0xFFFF;
    let mut sum2 = (rem * sum1) % BASE;
    sum1 += (a2 & 0xFFFF) + BASE - 1;
    sum2 += (a1 >> 16) + (a2 >> 16) + BASE - rem;
    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum1 >= BASE {
        sum1 -= BASE;
    }
    if sum2 >= BASE << 1 {
        sum2 -= BASE << 1;
    }
    if sum2 >= BASE {
        sum2 -= BASE;
    }
    sum1 | (sum2 << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Vec<Vec<u8>> {
        let mut seed = 0x1234_5678_u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as u8
        };
        let mut v: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![7],
            b"Hello world!".to_vec(),
            vec![0; 2048],
            vec![5; 2048],
            vec![0; 70_000],
            (0..=255u8).collect(),
        ];
        for len in [1usize, 7, 8, 9, 100, 4097] {
            v.push((0..len).map(|_| rnd()).collect());
            v.push((0..len).map(|_| if rnd() < 200 { 0 } else { rnd() }).collect());
        }
        v
    }

    fn zlib(data: &[u8], first: bool, last: bool) -> Vec<u8> {
        let mut v = compress_band(data, first, last);
        if last {
            v.extend_from_slice(&adler32(data).to_be_bytes());
        }
        v
    }

    #[test]
    fn single_band_matches_fdeflate_byte_for_byte() {
        for d in samples() {
            assert_eq!(zlib(&d, true, true), fdeflate::compress_to_vec(&d), "len {}", d.len());
        }
    }

    #[test]
    fn stitched_bands_decode_to_the_input() {
        for d in samples() {
            for pieces in [1usize, 2, 3, 5] {
                let size = d.len().div_ceil(pieces).max(1);
                let parts: Vec<&[u8]> = if d.is_empty() { vec![&d[..]] } else { d.chunks(size).collect() };
                let mut stream = Vec::new();
                let mut adler = adler32(parts[0]);
                for (i, p) in parts.iter().enumerate() {
                    stream.extend(compress_band(p, i == 0, i + 1 == parts.len()));
                    if i > 0 {
                        adler = adler32_combine(adler, adler32(p), p.len() as u64);
                    }
                }
                stream.extend_from_slice(&adler.to_be_bytes());
                assert_eq!(adler, adler32(&d));
                let back = fdeflate::decompress_to_vec(&stream)
                    .unwrap_or_else(|e| panic!("len {} in {pieces} bands: {e:?}", d.len()));
                assert_eq!(back, d, "len {} in {pieces} bands", d.len());
            }
        }
    }

    #[test]
    fn stored_bands_decode_to_the_input() {
        for d in samples().into_iter().chain([vec![9u8; 200_000]]) {
            let mut stream = Vec::new();
            let size = d.len().div_ceil(3).max(1);
            let parts: Vec<&[u8]> = if d.is_empty() { vec![&d[..]] } else { d.chunks(size).collect() };
            for (i, p) in parts.iter().enumerate() {
                stream.extend(stored_band(p, i == 0, i + 1 == parts.len()));
            }
            stream.extend_from_slice(&adler32(&d).to_be_bytes());
            if !d.is_empty() {
                assert_eq!(fdeflate::decompress_to_vec(&stream).unwrap(), d, "len {}", d.len());
            }
        }
    }

    #[test]
    fn adler_combine_matches_whole() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for cut in [0usize, 1, 5551, 65_521, 65_522, 123_457, 299_999, 300_000] {
            let (a, b) = data.split_at(cut);
            assert_eq!(adler32_combine(adler32(a), adler32(b), b.len() as u64), adler32(&data), "cut {cut}");
        }
    }
}
