use std::fmt;

/// Everything that can go wrong in a conversion.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Decoding or encoding an image failed.
    Image(image::ImageError),
    /// Reading or writing a file failed.
    Io(std::io::Error),
    /// An [`Options`](crate::Options) value is out of range; the message names it.
    InvalidOptions(String),
    /// The input cannot be converted: no pixels, or an output size beyond the supported limit.
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Image(e) => write!(f, "{e}"),
            Error::Io(e) => write!(f, "{e}"),
            Error::InvalidOptions(m) => write!(f, "invalid options: {m}"),
            Error::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

// `source` stays `None` because `Display` already includes the inner message; reporters that walk the chain
// would print it twice.
impl std::error::Error for Error {}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Image(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
