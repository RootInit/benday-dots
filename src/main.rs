use benday_dots::{chart, cli::Cli, pipeline};
use clap::Parser;

fn real_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let opts = cli.to_options()?;
    if let Some(path) = &cli.test_chart {
        let (w, h) = cli.chart_size;
        let (img, _) = chart::test_chart(w, h, &opts.inks);
        img.save(path)?;
        return Ok(());
    }
    let (Some(input), Some(output)) = (&cli.input, &cli.output) else { unreachable!("clap requires both paths") };
    let report = pipeline::run(input, output, &opts)?;
    if report.pitch_clamped {
        eprintln!("{}", pipeline::pitch_warning());
    }
    Ok(())
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
