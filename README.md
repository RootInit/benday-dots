# benday-dots

[![crates.io](https://img.shields.io/crates/v/benday-dots.svg)](https://crates.io/crates/benday-dots)
[![docs.rs](https://docs.rs/benday-dots/badge.svg)](https://docs.rs/benday-dots)
[![license](https://img.shields.io/crates/l/benday-dots.svg)](#license)

Turn photos into old comic book prints.

The image is flattened into a handful of colour regions using the 64 tints a 1960s comic printer actually had
(each of yellow, magenta and cyan at 0, 25, 50 or 100%). Each region is then screened as Ben-Day dots, one
angled dot grid per ink, on yellowed newsprint with a touch of misregistration. Black ink lines and solid
shadows are added on top. A pop-art preset swaps that for big 45° dots, a short flat palette and white paper.

It works as a command-line tool and as a library you can drop into an image pipeline.

Each row shows the original, `--style comic` and `--style pop`, all rendered with `--scale 3`.

![Lenna](https://raw.githubusercontent.com/RootInit/benday-dots/main/assets/lenna.jpg)
![Astronaut](https://raw.githubusercontent.com/RootInit/benday-dots/main/assets/astronaut.jpg)
![Chelsea the cat](https://raw.githubusercontent.com/RootInit/benday-dots/main/assets/chelsea.jpg)
![Coffee](https://raw.githubusercontent.com/RootInit/benday-dots/main/assets/coffee.jpg)

Detail at 1:1:

![Dots up close](https://raw.githubusercontent.com/RootInit/benday-dots/main/assets/lenna_detail.png)

## Install

```sh
cargo install benday-dots
```

## Command line

```sh
benday-dots photo.jpg print.png
benday-dots photo.jpg print.png --style pop
benday-dots photo.jpg print.png --dots-across 400 --scale 2 --paper white
benday-dots photo.jpg print.png --colors 8
```

Input can be JPEG or PNG; output is PNG. Any flag you pass overrides the style preset. A few worth knowing:

| Flag | What it does |
|---|---|
| `--style comic\|pop` | Look preset (default `comic`). |
| `--dots-across N` | Dot rows across the output width. Default 250 for comic, 70 for pop. |
| `--scale S` | Output size as a multiple of the input. Region shapes don't change, only the dots get finer. |
| `--colors N` | Pick N colours from the image with k-means instead of the full 64-tint chart. |
| `--inks aged\|process` | Faded newsprint inks or clean modern ones. |
| `--paper newsprint\|white\|cover` | Paper colour. |
| `--misregister`, `--dot-gain`, `--grain` | Printing imperfections; set to 0 for a clean print. |
| `--blacks L` | Lightness (0 to 100) below which areas go solid black. `0` turns this off. |
| `--no-lines` | Skip the ink outlines. |
| `--dump-stages DIR` | Save the intermediate images, handy for tuning. |

`benday-dots --help` has the rest.

## Library

```toml
[dependencies]
benday-dots = { version = "0.1", default-features = false }
```

With default features off you get only the in-memory conversion: no file I/O, no PNG/JPEG codecs and no `clap`.
You give it decoded pixels and get decoded pixels back.

```rust
use benday_dots::{convert, Options};
use benday_dots::image::RgbImage;

fn comicify(frame: &RgbImage) -> Result<RgbImage, benday_dots::Error> {
    convert(frame, &Options::comic().dots_across(180.0))
}
```

If your pipeline doesn't use the `image` crate, or uses a different version, pass raw bytes instead:

```rust
use benday_dots::{convert_raw, Options};

fn comicify(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, benday_dots::Error> {
    // Tightly packed RGB8 in and out; the output is `scale` times the input size.
    let (_w, _h, pixels) = convert_raw(width, height, rgb, &Options::pop())?;
    Ok(pixels)
}
```

There is also `convert_dynamic` for any `image::DynamicImage` (greyscale, 16-bit, RGBA and so on; alpha is
dropped), and `convert_with_stages`, which also returns the per-pixel tint map and the black ink mask.

To read and write files, keep the `io` feature:

```rust,no_run
use benday_dots::{convert_file, Options};

fn main() -> Result<(), benday_dots::Error> {
    convert_file("photo.jpg", "print.png", &Options::pop().dots_across(90.0))
}
```

### Features

- `io` (default): `convert_file`, plus JPEG and PNG support through `image`.
- `cli` (default, implies `io`): the `benday-dots` binary.

## How it works

1. Shrink the photo to a working size (2000 px on the long edge by default) and smooth it with an edge-preserving
   bilateral filter, so regions go flat but edges stay sharp.
2. Stretch the contrast, push the colours a bit (less so for near-greys, so skin and walls don't turn pink), and
   snap every pixel to the nearest printable tint. Clean the result up: remove speckles, merge tiny regions, and
   recolour each region from its average.
3. Find ink lines with an XDoG edge detector, and mark the darkest areas as solid black.
4. At full output resolution, screen each ink as round dots at its own angle and mix the inks on the paper
   colour, then lay the black on top.

All the region work happens on the working grid. A bigger `--scale` therefore costs more only in the final dot
pass.

A 12-megapixel photo takes about half a second on an 18-thread laptop CPU, including JPEG decode and PNG encode.
The smoothing has hand-written AVX2 code that is picked at run time, with a portable fallback for other CPUs.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

`src/deflate.rs` is adapted from [fdeflate](https://github.com/image-rs/fdeflate) (MIT OR Apache-2.0).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or
conditions.

### Example images

The examples are standard image-processing test images:

- Lenna (Lena Forsén), from the USC-SIPI image database. The photo was originally published by Playboy Enterprises,
  which holds its copyright.
- Astronaut Eileen Collins, NASA, public domain.
- Chelsea the cat, by Stéfan van der Walt, CC0.
- Coffee, by Rachel Michetti (courtesy of Pikolo Espresso Bar), CC0.

The last three come from [scikit-image](https://github.com/scikit-image/scikit-image).
