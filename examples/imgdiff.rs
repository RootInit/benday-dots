//! imgdiff A.png B.png: equivalence metrics between two same-size renders.
//! Prints MAD (mean abs channel diff), px>32 (% pixels with any channel off by more than 32), bMAD (MAD of
//! 16x16 block means, i.e. tone) and blocks>16 (% blocks whose mean is off by more than 16).
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let x = image::open(&a[0]).unwrap().to_rgb8();
    let y = image::open(&a[1]).unwrap().to_rgb8();
    assert_eq!(x.dimensions(), y.dimensions(), "size mismatch");
    let (w, h) = x.dimensions();
    let (xr, yr) = (x.as_raw(), y.as_raw());
    let mut sum = 0u64;
    let mut big = 0u64;
    for (p, q) in xr.as_chunks::<3>().0.iter().zip(yr.as_chunks::<3>().0) {
        let mut m = 0;
        for k in 0..3 {
            let d = (p[k] as i32 - q[k] as i32).abs();
            sum += d as u64;
            m = m.max(d);
        }
        if m > 32 {
            big += 1;
        }
    }
    let n = (w * h) as f64;
    let b = 16u32;
    let (bw, bh) = (w / b, h / b);
    let mut tsum = 0f64;
    let mut tbig = 0u64;
    for by in 0..bh {
        for bx in 0..bw {
            let mut s = [0f64; 6];
            for yy in 0..b {
                for xx in 0..b {
                    let i = (((by * b + yy) * w + bx * b + xx) * 3) as usize;
                    for k in 0..3 {
                        s[k] += xr[i + k] as f64;
                        s[3 + k] += yr[i + k] as f64;
                    }
                }
            }
            let mut m = 0f64;
            for k in 0..3 {
                let d = (s[k] - s[3 + k]).abs() / (b * b) as f64;
                tsum += d;
                m = m.max(d);
            }
            if m > 16.0 {
                tbig += 1;
            }
        }
    }
    println!(
        "MAD {:.3} px>32 {:.3} bMAD {:.3} blocks>16 {:.3}",
        sum as f64 / (n * 3.0),
        100.0 * big as f64 / n,
        tsum / (bw * bh * 3) as f64,
        100.0 * tbig as f64 / (bw * bh) as f64
    );
}
