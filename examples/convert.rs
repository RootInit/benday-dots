//! Usage: convert INPUT OUTPUT [pop]
use benday_dots::{Options, convert_file};

fn main() -> Result<(), benday_dots::Error> {
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        eprintln!("usage: convert INPUT OUTPUT [pop]");
        std::process::exit(2);
    };
    let opts = if args.next().as_deref() == Some("pop") { Options::pop() } else { Options::comic() };
    convert_file(input, output, &opts)
}
