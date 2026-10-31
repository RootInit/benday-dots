use crate::codes::POP_CODES;
use crate::error::Error;
use crate::ink::{PAPER_COVER, PAPER_NEWSPRINT, PAPER_WHITE};
#[cfg(feature = "io")]
use crate::io::PngCompression;
use crate::types::InkSet;
#[cfg(feature = "io")]
use std::path::PathBuf;

/// Smallest dot pitch in output pixels; smaller requests are raised to it.
pub const MIN_PITCH: f32 = 6.0;
/// Largest accepted dot pitch in output pixels. Beyond this the tint maths overflows.
pub const MAX_PITCH: f32 = 10_000.0;
/// Most bilateral passes accepted.
const MAX_BILATERAL_PASSES: u32 = 32;
/// Largest accepted misregistration, as a fraction of the pitch.
const MAX_MISREGISTER: f32 = 100.0;
/// Codes are `0..CODE_COUNT`.
const CODE_COUNT: u8 = 64;

/// Ink swatch set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum InkChoice {
    /// Slightly muted, aged-newsprint inks.
    Aged,
    /// Saturated modern process inks.
    Process,
}

/// Paper colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum PaperChoice {
    /// Yellowed newsprint.
    Newsprint,
    /// Pure white.
    White,
    /// Bright cover stock.
    Cover,
}

/// Look preset; see [`Options::from_style`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum Style {
    /// Silver-Age comic: fine dots, aged newsprint, misregistration and grain.
    Comic,
    /// Pop-art: large 45 degree dots, a small flat palette, white paper, clean registration.
    Pop,
}

/// Everything that controls a conversion.
///
/// Start from a preset ([`Options::comic`], [`Options::pop`] or [`Options::from_style`]) and adjust it with the
/// chainable setters. Values are checked by [`Options::validate`] when a conversion runs.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// Output size is input size times this; must be positive.
    pub scale: f32,
    /// Fixed dot pitch in output pixels; overrides `dots_across`. Pitches below [`MIN_PITCH`] are raised to it.
    pub pitch: Option<f32>,
    /// Dots across the output width (pitch = width / this).
    pub dots_across: f32,
    /// `Some(k)` selects k-means palette mode with `k` colours (1 to 64).
    pub colors: Option<usize>,
    /// Ink colours and paper.
    pub inks: InkSet,
    /// Per-ink misregistration offset as a fraction of the pitch.
    pub misregister: f32,
    /// Signed dot radius change in output pixels.
    pub dot_gain: f32,
    /// Paper grain and ink mottle amplitude; 0 disables.
    pub grain: f32,
    /// Seed for misregistration direction, phase, grain and k-means.
    pub seed: u64,
    /// Spotted-black threshold as an input L* value (0 to 100). Pixels darker than it, after the automatic
    /// levels stretch, become solid black, and the same value sets the line threshold.
    ///
    /// `None` picks the level from the image (the 8th percentile of L*, at most 15) and keeps the style's
    /// solid-black floor, so near-black areas are always inked. An explicit value replaces both and can be lower
    /// than the floor. `Some(v)` with `v <= 0.0` turns the spotted-black mask off; lines are then drawn
    /// wherever the line detector goes negative.
    pub blacks: Option<f32>,
    /// Draw XDoG ink lines.
    pub lines: bool,
    /// Long edge of the internal working grid, in pixels.
    pub work_size: u32,
    /// PNG encoder effort for [`convert_file`](crate::convert_file).
    #[cfg(feature = "io")]
    pub png_compression: PngCompression,
    /// Write intermediate stage PNGs into this directory.
    #[cfg(feature = "io")]
    pub dump_dir: Option<PathBuf>,
    /// Print per-stage timings and warnings to stderr.
    pub verbose: bool,
    #[doc(hidden)]
    pub boost: bool,
    #[doc(hidden)]
    pub l_weight: f32,
    #[doc(hidden)]
    pub hue_weight: f32,
    #[doc(hidden)]
    pub boost_chroma: f32,
    #[doc(hidden)]
    pub bilateral_passes: u32,
    #[doc(hidden)]
    pub mode_radius_mul: f32,
    #[doc(hidden)]
    pub min_area_mul: f32,
    #[doc(hidden)]
    pub line_dilate_mul: f32,
    #[doc(hidden)]
    pub line_area_mul: f32,
    #[doc(hidden)]
    pub angles: [f32; 3],
    #[doc(hidden)]
    pub repick_mul: f32,
    #[doc(hidden)]
    pub allowed: Option<Vec<u8>>,
    /// Keep the near-black full-overprint codes in the palette instead of leaving those tones to the black mask.
    #[doc(hidden)]
    pub keep_dark_codes: bool,
    /// Stretched L* below which a pixel is solid black when `blacks` is automatic.
    #[doc(hidden)]
    pub solid_black_l: f32,
}

impl Default for Options {
    fn default() -> Self {
        Options::comic()
    }
}

impl Options {
    /// The Silver-Age comic preset (the default).
    pub fn comic() -> Self {
        Self::from_style(Style::Comic)
    }

    /// The pop-art preset.
    pub fn pop() -> Self {
        Self::from_style(Style::Pop)
    }

    /// Preset defaults for `style`.
    pub fn from_style(style: Style) -> Self {
        let pop = style == Style::Pop;
        Options {
            scale: 1.0,
            pitch: None,
            dots_across: if pop { 70.0 } else { 250.0 },
            colors: None,
            inks: if pop {
                InkSet::process().with_paper(PAPER_WHITE)
            } else {
                InkSet::aged().with_paper(PAPER_NEWSPRINT)
            },
            misregister: if pop { 0.0 } else { 0.15 },
            dot_gain: if pop { 0.0 } else { 0.25 },
            grain: if pop { 0.0 } else { 0.04 },
            seed: 1,
            blacks: None,
            lines: true,
            work_size: 2000,
            #[cfg(feature = "io")]
            png_compression: PngCompression::default(),
            #[cfg(feature = "io")]
            dump_dir: None,
            verbose: false,
            boost: true,
            l_weight: if pop { 0.9 } else { 1.5 },
            hue_weight: if pop { 1.0 } else { 0.0 },
            boost_chroma: 1.3,
            bilateral_passes: if pop { 5 } else { 3 },
            mode_radius_mul: if pop { 2.0 } else { 1.0 },
            min_area_mul: if pop { 2.0 } else { 1.0 },
            line_dilate_mul: if pop { 2.5 } else { 1.0 },
            line_area_mul: if pop { 4.0 } else { 1.0 },
            repick_mul: 6.0,
            angles: if pop { [45.0; 3] } else { crate::screen::BENDAY_ANGLES },
            allowed: pop.then(|| POP_CODES.to_vec()),
            keep_dark_codes: pop,
            solid_black_l: if pop { 14.0 } else { 25.0 },
        }
    }

    /// Output size as a multiple of the input size.
    #[must_use]
    pub fn scale(mut self, v: f32) -> Self {
        self.scale = v;
        self
    }

    /// Dots across the output width.
    #[must_use]
    pub fn dots_across(mut self, v: f32) -> Self {
        self.dots_across = v;
        self
    }

    /// Fixed dot pitch in output pixels, overriding `dots_across`; `None` goes back to `dots_across`.
    #[must_use]
    pub fn pitch(mut self, v: impl Into<Option<f32>>) -> Self {
        self.pitch = v.into();
        self
    }

    /// Choose the ink set, keeping the current paper.
    #[must_use]
    pub fn inks(mut self, v: InkChoice) -> Self {
        let paper = self.inks.paper;
        self.inks = match v {
            InkChoice::Aged => InkSet::aged(),
            InkChoice::Process => InkSet::process(),
        }
        .with_paper(paper);
        self
    }

    /// Choose the paper colour.
    #[must_use]
    pub fn paper(mut self, v: PaperChoice) -> Self {
        self.inks.paper = match v {
            PaperChoice::Newsprint => PAPER_NEWSPRINT,
            PaperChoice::White => PAPER_WHITE,
            PaperChoice::Cover => PAPER_COVER,
        };
        self
    }

    /// Misregistration offset as a fraction of the pitch.
    #[must_use]
    pub fn misregister(mut self, v: f32) -> Self {
        self.misregister = v;
        self
    }

    /// Signed dot radius change in output pixels.
    #[must_use]
    pub fn dot_gain(mut self, v: f32) -> Self {
        self.dot_gain = v;
        self
    }

    /// Paper grain amplitude; 0 disables.
    #[must_use]
    pub fn grain(mut self, v: f32) -> Self {
        self.grain = v;
        self
    }

    /// Seed for all pseudo-random choices.
    #[must_use]
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Spotted-black input L* threshold; `None` is automatic and a value of 0 or less disables the spotted-black
    /// mask. An explicit value replaces the automatic level and its near-black floor, so it can ink less than
    /// automatic does (see the `blacks` field).
    #[must_use]
    pub fn blacks(mut self, v: impl Into<Option<f32>>) -> Self {
        self.blacks = v.into();
        self
    }

    /// Enable or disable XDoG ink lines.
    #[must_use]
    pub fn lines(mut self, v: bool) -> Self {
        self.lines = v;
        self
    }

    /// Use the full 64-code chart (`None`) or k-means with `n` colours (1 to 64).
    #[must_use]
    pub fn colors(mut self, n: Option<usize>) -> Self {
        self.colors = n;
        self
    }

    /// Long edge of the working grid in pixels.
    #[must_use]
    pub fn work_size(mut self, v: u32) -> Self {
        self.work_size = v;
        self
    }

    /// PNG encoder effort used when writing a `.png` file.
    #[cfg(feature = "io")]
    #[must_use]
    pub fn png_compression(mut self, v: PngCompression) -> Self {
        self.png_compression = v;
        self
    }

    /// Write intermediate stage PNGs into `dir`.
    #[cfg(feature = "io")]
    #[must_use]
    pub fn dump_stages(mut self, dir: impl Into<PathBuf>) -> Self {
        self.dump_dir = Some(dir.into());
        self
    }

    /// Print timings and warnings to stderr.
    #[must_use]
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Check every value is in range.
    ///
    /// # Errors
    /// [`Error::InvalidOptions`] naming the first value that is non-finite, out of range or, for the hidden
    /// `allowed` list, empty or holding a code of 64 or more.
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |m: String| Err(Error::InvalidOptions(m));
        let positive = |name: &str, v: f32| -> Result<(), Error> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidOptions(format!("{name} must be positive, got {v}")))
            }
        };
        let non_negative = |name: &str, v: f32| -> Result<(), Error> {
            if v.is_finite() && v >= 0.0 {
                Ok(())
            } else {
                Err(Error::InvalidOptions(format!("{name} must be non-negative, got {v}")))
            }
        };
        positive("scale", self.scale)?;
        positive("dots_across", self.dots_across)?;
        if let Some(p) = self.pitch {
            positive("pitch", p)?;
            if p > MAX_PITCH {
                return bad(format!("pitch must be at most {MAX_PITCH}, got {p}"));
            }
        }
        if let Some(k) = self.colors
            && !(1..=usize::from(CODE_COUNT)).contains(&k)
        {
            return bad(format!("colors must be in 1..={CODE_COUNT}, got {k}"));
        }
        if self.work_size == 0 {
            return bad("work_size must be positive".into());
        }
        if !(self.misregister.is_finite() && self.misregister.abs() <= MAX_MISREGISTER) {
            return bad(format!("misregister must be within +-{MAX_MISREGISTER}, got {}", self.misregister));
        }
        if !self.dot_gain.is_finite() {
            return bad(format!("dot_gain must be finite, got {}", self.dot_gain));
        }
        non_negative("grain", self.grain)?;
        if let Some(b) = self.blacks {
            non_negative("blacks", b)?;
        }
        non_negative("l_weight", self.l_weight)?;
        non_negative("hue_weight", self.hue_weight)?;
        non_negative("boost_chroma", self.boost_chroma)?;
        non_negative("mode_radius_mul", self.mode_radius_mul)?;
        non_negative("min_area_mul", self.min_area_mul)?;
        non_negative("line_dilate_mul", self.line_dilate_mul)?;
        non_negative("line_area_mul", self.line_area_mul)?;
        non_negative("repick_mul", self.repick_mul)?;
        non_negative("solid_black_l", self.solid_black_l)?;
        if self.bilateral_passes > MAX_BILATERAL_PASSES {
            return bad(format!(
                "bilateral_passes must be at most {MAX_BILATERAL_PASSES}, got {}",
                self.bilateral_passes
            ));
        }
        if self.angles.iter().any(|a| !a.is_finite()) {
            return bad("angles must be finite".into());
        }
        if let Some(allowed) = &self.allowed {
            if allowed.is_empty() {
                return bad("allowed must list at least one code".into());
            }
            if let Some(c) = allowed.iter().find(|&&c| c >= CODE_COUNT) {
                return bad(format!("allowed codes must be below {CODE_COUNT}, got {c}"));
            }
        }
        Ok(())
    }

    /// Pitch for an output width and whether it had to be raised to [`MIN_PITCH`].
    #[doc(hidden)]
    pub fn resolve_pitch(&self, out_w: u32) -> (f32, bool) {
        let p = self.pitch.unwrap_or(out_w as f32 / self.dots_across);
        if p < MIN_PITCH { (MIN_PITCH, true) } else { (p, false) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pitch_rules() {
        let mut o = Options::comic();
        assert!((o.resolve_pitch(4032).0 - 16.128).abs() < 1e-3);
        o.pitch = Some(12.0);
        assert_eq!(o.resolve_pitch(4032), (12.0, false));
        o.pitch = Some(2.0);
        assert_eq!(o.resolve_pitch(4032), (6.0, true));
        o.pitch = None;
        assert_eq!(o.resolve_pitch(300), (6.0, true));
    }

    #[test]
    fn presets_differ() {
        let (c, p) = (Options::comic(), Options::pop());
        assert_eq!((c.dots_across, p.dots_across), (250.0, 70.0));
        assert_eq!(p.inks.paper, PAPER_WHITE);
        assert_eq!(p.allowed.as_deref(), Some(&POP_CODES[..]));
        assert!(p.keep_dark_codes && !c.keep_dark_codes);
        assert!(p.solid_black_l < c.solid_black_l);
    }

    #[test]
    fn setters_keep_paper_across_ink_change() {
        let o = Options::pop().paper(PaperChoice::Cover).inks(InkChoice::Aged);
        assert_eq!(o.inks.paper, PAPER_COVER);
    }

    #[test]
    fn setters_can_unset_pitch_and_blacks() {
        let o = Options::comic().pitch(9.0).blacks(12.0);
        assert_eq!((o.pitch, o.blacks), (Some(9.0), Some(12.0)));
        let o = o.pitch(None).blacks(None);
        assert_eq!((o.pitch, o.blacks), (None, None));
    }

    #[test]
    fn validation_rejects_bad_values() {
        assert!(Options::comic().validate().is_ok());
        assert!(Options::pop().validate().is_ok());
        assert!(Options::comic().scale(0.0).validate().is_err());
        assert!(Options::comic().dots_across(-1.0).validate().is_err());
        assert!(Options::comic().colors(Some(65)).validate().is_err());
        assert!(Options::comic().colors(Some(0)).validate().is_err());
        assert!(Options::comic().colors(Some(64)).validate().is_ok());
        assert!(Options::comic().grain(f32::NAN).validate().is_err());
        assert!(Options::comic().work_size(0).validate().is_err());
        assert!(Options::comic().pitch(1e30).validate().is_err());
        assert!(Options::comic().pitch(MAX_PITCH).validate().is_ok());
        assert!(Options::comic().misregister(1e30).validate().is_err());
    }

    #[test]
    fn validation_rejects_bad_hidden_knobs() {
        let with = |f: fn(&mut Options)| {
            let mut o = Options::comic();
            f(&mut o);
            o.validate()
        };
        assert!(with(|o| o.allowed = Some(vec![])).is_err());
        assert!(with(|o| o.allowed = Some(vec![64])).is_err());
        assert!(with(|o| o.allowed = Some(vec![200])).is_err());
        assert!(with(|o| o.allowed = Some(vec![0, 63])).is_ok());
        assert!(with(|o| o.l_weight = f32::NAN).is_err());
        assert!(with(|o| o.hue_weight = -1.0).is_err());
        assert!(with(|o| o.angles = [0.0, f32::INFINITY, 0.0]).is_err());
        assert!(with(|o| o.bilateral_passes = 1_000_000).is_err());
        assert!(with(|o| o.repick_mul = f32::NAN).is_err());
        assert!(with(|o| o.mode_radius_mul = -1.0).is_err());
        assert!(with(|o| o.boost_chroma = f32::INFINITY).is_err());
    }
}
