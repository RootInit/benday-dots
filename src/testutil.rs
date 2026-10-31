//! Helpers for the `#[ignore]`d benchmarks. CPU time is reported next to wall time because wall time means little
//! on a loaded machine.

/// CPU time of all threads in ms.
pub(crate) fn cpu_ms() -> f64 {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("schedstat")).ok())
        .map(|s| s.split_whitespace().next().unwrap().parse::<f64>().unwrap())
        .sum::<f64>()
        / 1e6
}

/// Best of `reps` runs as (wall ms, cpu ms).
pub(crate) fn best_of<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, f64) {
    let (mut bw, mut bc) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        let (c0, t) = (cpu_ms(), std::time::Instant::now());
        std::hint::black_box(f());
        bw = bw.min(t.elapsed().as_secs_f64() * 1e3);
        bc = bc.min(cpu_ms() - c0);
    }
    (bw, bc)
}

/// Best of 7 as (wall ms, cpu ms).
pub(crate) fn best<T>(f: impl FnMut() -> T) -> (f64, f64) {
    best_of(7, f)
}

/// Run `f` once, print its wall and CPU time under `label`, and return its result.
pub(crate) fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let (c0, t) = (cpu_ms(), std::time::Instant::now());
    let r = f();
    eprintln!("{label}: wall {:.1} ms, cpu {:.1} ms", t.elapsed().as_secs_f64() * 1e3, cpu_ms() - c0);
    r
}
