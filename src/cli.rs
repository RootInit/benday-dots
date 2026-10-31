use crate::error::Error;
use crate::io::PngCompression;
use crate::options::{InkChoice, Options, PaperChoice, Style};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// How colours are chosen for the flat regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PaletteMode {
    /// Match against the full 64-code Silver-Age chart.
    Benday,
    /// Cluster the image with k-means and snap the centroids to chart codes.
    Kmeans,
}

/// Colours used by `--palette kmeans` when `--colors` is not given.
const DEFAULT_COLORS: usize = 12;

fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let bad = || format!("bad size '{s}', expected WxH");
    let (w, h) = s.split_once(['x', 'X']).ok_or_else(bad)?;
    let parse = |v: &str| v.trim().parse::<u32>().ok().filter(|&n| n > 0);
    match (parse(w), parse(h)) {
        (Some(w), Some(h)) => Ok((w, h)),
        _ => Err(bad()),
    }
}

/// Turn a photo into a Silver-Age comic print with Ben-Day dots.
#[derive(Parser, Debug, Clone)]
#[command(name = "benday-dots", version)]
pub struct Cli {
    /// Input image (jpeg/png)
    #[arg(required_unless_present = "test_chart")]
    pub input: Option<PathBuf>,

    /// Output image (png)
    #[arg(required_unless_present = "test_chart")]
    pub output: Option<PathBuf>,

    /// Look preset; explicit flags override its defaults
    #[arg(long, value_enum, default_value_t = Style::Comic)]
    pub style: Style,

    /// Dots across the output width; pitch = W_out / N [default: 250, pop 70]
    #[arg(long)]
    pub dots_across: Option<f32>,

    /// Dot pitch in output px; overrides --dots-across
    #[arg(long)]
    pub pitch: Option<f32>,

    /// Output size = input size x scale
    #[arg(long, default_value_t = 1.0)]
    pub scale: f32,

    /// 64-code chart, or k-means centroids snapped to codes [default: benday, or kmeans when --colors is given]
    #[arg(long, value_enum)]
    pub palette: Option<PaletteMode>,

    /// Number of k-means colours, 1 to 64; implies --palette kmeans [default with it: 12]
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub colors: Option<u8>,

    /// Ink swatch set [default: aged, pop process]
    #[arg(long, value_enum)]
    pub inks: Option<InkChoice>,

    /// Paper colour [default: newsprint, pop white]
    #[arg(long, value_enum)]
    pub paper: Option<PaperChoice>,

    /// Per-ink misregistration offset, as a fraction of the pitch [default: 0.15, pop 0]
    #[arg(long)]
    pub misregister: Option<f32>,

    /// Signed dot radius change in output px [default: 0.25, pop 0]
    #[arg(long, allow_negative_numbers = true)]
    pub dot_gain: Option<f32>,

    /// Paper grain / ink mottle amplitude, 0 disables [default: 0.04, pop 0]
    #[arg(long)]
    pub grain: Option<f32>,

    /// Seed for misregister direction, phase, grain and k-means
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Spotted-black input L* threshold. Auto (the default) uses min(15, 8th percentile of L*) and always inks
    /// near-black areas; an explicit value replaces both and may be lower. 0 disables the spotted-black mask
    #[arg(long)]
    pub blacks: Option<f32>,

    /// Disable XDoG ink lines (spotted blacks unaffected)
    #[arg(long)]
    pub no_lines: bool,

    /// Working-grid long edge
    #[arg(long, default_value_t = 2000)]
    pub work_size: u32,

    /// PNG encoder effort: fast (smallest files), fastest, or none (stored, about 3x larger)
    #[arg(long, value_enum, default_value_t = PngCompression::Fast)]
    pub png: PngCompression,

    /// Write intermediate PNGs into this directory
    #[arg(long)]
    pub dump_stages: Option<PathBuf>,

    /// Per-stage timings to stderr
    #[arg(short, long)]
    pub verbose: bool,

    /// Write the synthetic test chart and exit
    #[arg(long, conflicts_with_all = ["input", "output"])]
    pub test_chart: Option<PathBuf>,

    /// Test chart size, WxH
    #[arg(long, default_value = "1600x1200", value_parser = parse_size)]
    pub chart_size: (u32, u32),
}

impl Cli {
    /// Style preset, then explicit flags on top.
    ///
    /// # Errors
    /// [`Error::InvalidOptions`] when `--palette benday` is combined with `--colors`.
    pub fn to_options(&self) -> Result<Options, Error> {
        let mut o = Options::from_style(self.style);
        o.scale = self.scale;
        o.pitch = self.pitch;
        if let Some(v) = self.dots_across {
            o.dots_across = v;
        }
        o.colors = match (self.palette, self.colors) {
            (Some(PaletteMode::Benday), Some(_)) => {
                return Err(Error::InvalidOptions("--colors needs --palette kmeans, not benday".into()));
            }
            (Some(PaletteMode::Kmeans), n) => Some(n.map_or(DEFAULT_COLORS, usize::from)),
            (None, Some(n)) => Some(usize::from(n)),
            (_, None) => None,
        };
        if let Some(v) = self.inks {
            o = o.inks(v);
        }
        if let Some(v) = self.paper {
            o = o.paper(v);
        }
        if let Some(v) = self.misregister {
            o.misregister = v;
        }
        if let Some(v) = self.dot_gain {
            o.dot_gain = v;
        }
        if let Some(v) = self.grain {
            o.grain = v;
        }
        o.seed = self.seed;
        o.blacks = self.blacks;
        o.lines = !self.no_lines;
        o.work_size = self.work_size;
        o.png_compression = self.png;
        o.dump_dir.clone_from(&self.dump_stages);
        o.verbose = self.verbose;
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::POP_CODES;

    fn opts(args: &[&str]) -> Result<Options, Error> {
        let mut v = vec!["benday-dots", "a", "b"];
        v.extend_from_slice(args);
        Cli::try_parse_from(v).unwrap().to_options()
    }

    #[test]
    fn defaults_parse() {
        let c = Cli::try_parse_from(["benday-dots", "a.jpg", "b.png"]).unwrap();
        assert_eq!(c.dots_across, None);
        assert_eq!(c.style, Style::Comic);
        assert_eq!(c.palette, None);
        assert_eq!(c.work_size, 2000);
        assert_eq!(c.chart_size, (1600, 1200));
        assert!(!c.no_lines);
        let o = c.to_options().unwrap();
        assert_eq!((o.dots_across, o.colors), (250.0, None));
        assert_eq!(o.png_compression, PngCompression::Fast);
    }

    #[test]
    fn test_chart_rejects_positional_paths() {
        assert!(Cli::try_parse_from(["benday-dots", "a.jpg", "b.png", "--test-chart", "c.png"]).is_err());
    }

    #[test]
    fn positionals_required_unless_chart() {
        assert!(Cli::try_parse_from(["benday-dots"]).is_err());
        let c = Cli::try_parse_from(["benday-dots", "--test-chart", "c.png"]).unwrap();
        assert!(c.input.is_none() && c.test_chart.is_some());
    }

    #[test]
    fn negative_dot_gain() {
        let c = Cli::try_parse_from(["benday-dots", "a", "b", "--dot-gain", "-0.5"]).unwrap();
        assert_eq!(c.dot_gain, Some(-0.5));
    }

    #[test]
    fn explicit_flag_overrides_pop_preset() {
        let pop = opts(&["--style", "pop"]).unwrap();
        assert_eq!(pop.dots_across, 70.0);
        assert_eq!(pop.angles, [45.0; 3]);
        assert_eq!(pop.allowed.as_deref(), Some(&POP_CODES[..]));
        let cfg = opts(&["--style", "pop", "--dots-across", "120", "--grain", "0.1"]).unwrap();
        assert_eq!((cfg.dots_across, cfg.grain, cfg.misregister), (120.0, 0.1, 0.0));
    }

    #[test]
    fn colors_are_range_checked_and_imply_kmeans() {
        for bad in ["0", "65", "99", "-1"] {
            assert!(Cli::try_parse_from(["benday-dots", "a", "b", "--colors", bad]).is_err(), "{bad}");
        }
        assert_eq!(opts(&["--colors", "64"]).unwrap().colors, Some(64));
        assert_eq!(opts(&["--colors", "8"]).unwrap().colors, Some(8));
        assert_eq!(opts(&["--palette", "kmeans"]).unwrap().colors, Some(DEFAULT_COLORS));
        assert_eq!(opts(&["--palette", "kmeans", "--colors", "5"]).unwrap().colors, Some(5));
        assert_eq!(opts(&["--palette", "benday"]).unwrap().colors, None);
        assert!(opts(&["--palette", "benday", "--colors", "5"]).is_err());
    }

    #[test]
    fn mapping_of_the_remaining_flags() {
        let o = opts(&["--paper", "white", "--no-lines", "--png", "none", "--blacks", "7"]).unwrap();
        assert_eq!(o.inks.paper, crate::ink::PAPER_WHITE);
        assert!(!o.lines && o.boost);
        assert_eq!((o.png_compression, o.blacks), (PngCompression::None, Some(7.0)));
    }

    #[test]
    fn chart_size_is_validated_by_clap() {
        let c = Cli::try_parse_from(["benday-dots", "--test-chart", "c.png", "--chart-size", "80X60"]).unwrap();
        assert_eq!(c.chart_size, (80, 60));
        for bad in ["80", "0x5", "ax5", "80x"] {
            assert!(
                Cli::try_parse_from(["benday-dots", "--test-chart", "c.png", "--chart-size", bad]).is_err(),
                "{bad}"
            );
        }
    }
}
