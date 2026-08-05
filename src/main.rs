#[cfg(not(target_os = "linux"))]
compile_error!("kdeocr only supports Linux.");

mod keyboard;
mod models;
mod ocr;

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Args, CommandFactory, Parser, Subcommand};
use tempfile::TempDir;
use thiserror::Error;

const EXIT_OK: u8 = 0;
const EXIT_CANCELLED: u8 = 2;
const EXIT_MISSING_DEPENDENCY: u8 = 3;
const EXIT_INVALID_INPUT: u8 = 4;
const EXIT_OCR: u8 = 5;
const EXIT_OUTPUT: u8 = 6;

const CLI_STYLES: Styles = Styles::styled()
    .header(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Cyan.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Yellow.on_default());

#[derive(Debug, Parser)]
#[command(
    name = "kocr",
    version,
    propagate_version = true,
    about = "Offline OCR tool for KDE Plasma",
    color = clap::ColorChoice::Always,
    styles = CLI_STYLES,
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CommandKind>,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    /// Capture a screen region
    Capture(CaptureArgs),

    /// Check required dependencies
    Doctor,

    /// List available models
    List,

    /// Install a model by ID or name
    Install(models::ProfileArgs),

    /// Uninstall a model by ID or name
    Uninstall(models::ProfileArgs),

    /// Select the active OCR model
    Use(models::ProfileArgs),

    /// Open the configuration file
    Config,

    /// Run the global shortcut daemon
    Daemon,

    /// Recognize text from an image
    Image(ocr::ImageArgs),

    /// Show command help
    #[command(name = "help", about = "Show command help")]
    Help,

    /// Show version
    Version,
}

#[derive(Debug, Args)]
struct CaptureArgs {
    /// Output path (optional)
    #[arg(value_name = "PNG")]
    output: Option<PathBuf>,

    /// Recognize the capture and copy text
    #[arg(short = 'o', long = "ocr")]
    ocr: bool,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("Capture cancelled")]
    Cancelled,

    #[error("Missing dependency: {0}")]
    MissingDependency(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model failed: {0}")]
    Model(#[from] models::ModelError),

    #[error("Config failed: {0}")]
    Config(models::ModelError),

    #[error("OCR failed: {0}")]
    Ocr(#[from] ocr::OcrError),

    #[error("Keyboard failed: {0}")]
    Keyboard(#[from] keyboard::KeyboardError),

    #[error("Output failed: {0}")]
    Output(String),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::from(EXIT_OK),
        Err(AppError::Cancelled) => ExitCode::from(EXIT_CANCELLED),
        Err(AppError::Model(error)) => {
            let error = AppError::Model(error);
            eprintln!("\x1b[31m{error}\x1b[0m");
            ExitCode::from(EXIT_INVALID_INPUT)
        }
        Err(AppError::Config(error)) => {
            let error = AppError::Config(error);
            eprintln!("\x1b[31m{error}\x1b[0m");
            ExitCode::from(EXIT_OUTPUT)
        }
        Err(AppError::Ocr(error)) => {
            let error = AppError::Ocr(error);
            eprintln!("\x1b[31m{error}\x1b[0m");
            ExitCode::from(EXIT_OCR)
        }
        Err(AppError::Keyboard(error)) => {
            let error = AppError::Keyboard(error);
            eprintln!("\x1b[31m{error}\x1b[0m");
            ExitCode::from(EXIT_OUTPUT)
        }
        Err(error) => {
            eprintln!("kocr: {error}");
            ExitCode::from(error_code(&error))
        }
    }
}

fn run(cli: Cli) -> Result<(), AppError> {
    match cli.command {
        Some(CommandKind::Capture(args)) => run_capture(args.output, args.ocr),
        Some(CommandKind::Doctor) => run_doctor(),
        Some(CommandKind::List) => models::list().map_err(AppError::Model),
        Some(CommandKind::Install(args)) => models::install(&args.profile).map_err(AppError::Model),
        Some(CommandKind::Uninstall(args)) => {
            models::uninstall(&args.profile).map_err(AppError::Model)
        }
        Some(CommandKind::Use(args)) => models::use_model(&args.profile).map_err(AppError::Model),
        Some(CommandKind::Config) => models::edit_config().map_err(AppError::Config),
        Some(CommandKind::Daemon) => keyboard::run(|| run_capture(None, true)).map_err(Into::into),
        Some(CommandKind::Image(args)) => ocr::run(args.image).map_err(AppError::Ocr),
        Some(CommandKind::Help) => print_help(),
        Some(CommandKind::Version) => {
            println!("kocr {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        None => print_help(),
    }
}

fn print_help() -> Result<(), AppError> {
    Cli::command()
        .print_help()
        .map_err(|error| AppError::Output(format!("could not print help: {error}")))?;
    println!();
    Ok(())
}

fn error_code(error: &AppError) -> u8 {
    match error {
        AppError::Cancelled => EXIT_CANCELLED,
        AppError::MissingDependency(_) => EXIT_MISSING_DEPENDENCY,
        AppError::InvalidInput(_) | AppError::Model(_) => EXIT_INVALID_INPUT,
        AppError::Config(_) => EXIT_OUTPUT,
        AppError::Ocr(_) => EXIT_OCR,
        AppError::Keyboard(_) | AppError::Output(_) => EXIT_OUTPUT,
    }
}

fn run_capture(output: Option<PathBuf>, recognize: bool) -> Result<(), AppError> {
    let spectacle = require_command("spectacle")?;
    let temp_dir = TempDir::new().map_err(|error| {
        AppError::Output(format!("could not create temporary directory: {error}"))
    })?;
    let capture_path = output
        .clone()
        .unwrap_or_else(|| temp_dir.path().join("capture.png"));

    let status = Command::new(spectacle)
        .args(spectacle_args())
        .arg(&capture_path)
        .stderr(Stdio::null())
        .status()
        .map_err(|error| AppError::Output(format!("could not start Spectacle: {error}")))?;

    if !capture_path.is_file() {
        if status.success() {
            return Err(AppError::Cancelled);
        }
        return Err(AppError::Output(format!(
            "Spectacle exited with {status} and produced no screenshot"
        )));
    }
    if !status.success() {
        return Err(AppError::Output(format!("Spectacle exited with {status}")));
    }

    let png = fs::read(&capture_path).map_err(|error| {
        AppError::InvalidInput(format!(
            "could not read {}: {error}",
            capture_path.display()
        ))
    })?;
    validate_png(&png)?;

    if recognize {
        let text = ocr::recognize(capture_path)?;
        copy_text(&text)?;
        println!("{text}");
    } else {
        copy_png(&png)?;
    }
    if let Some(path) = output {
        println!("{}", path.display());
    }
    Ok(())
}

fn spectacle_args() -> [&'static str; 5] {
    [
        "--region",
        "--background",
        "--nonotify",
        "--release-capture",
        "--output",
    ]
}

fn require_command(name: &str) -> Result<PathBuf, AppError> {
    find_command(name)
        .ok_or_else(|| AppError::MissingDependency(format!("{name} not found in PATH")))
}

fn validate_png(bytes: &[u8]) -> Result<(), AppError> {
    const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() <= PNG_SIGNATURE.len() {
        return Err(AppError::InvalidInput(
            "captured PNG is empty or truncated".to_owned(),
        ));
    }
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Err(AppError::InvalidInput(
            "captured file does not have a PNG signature".to_owned(),
        ));
    }
    Ok(())
}

fn copy_png(png: &[u8]) -> Result<(), AppError> {
    copy_clipboard(png, "image/png")
}

fn copy_text(text: &str) -> Result<(), AppError> {
    copy_clipboard(text.as_bytes(), "text/plain;charset=utf-8")
}

fn copy_clipboard(content: &[u8], mime: &str) -> Result<(), AppError> {
    let wl_copy = require_command("wl-copy")?;
    let mut child = Command::new(wl_copy)
        .args(["--type", mime])
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| AppError::Output(format!("could not start wl-copy: {error}")))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| AppError::Output("wl-copy stdin was not available".to_owned()))?;
    stdin
        .write_all(content)
        .map_err(|error| AppError::Output(format!("could not write to wl-copy: {error}")))?;
    drop(stdin);

    let status = child
        .wait()
        .map_err(|error| AppError::Output(format!("could not wait for wl-copy: {error}")))?;
    if !status.success() {
        return Err(AppError::Output(format!("wl-copy exited with {status}")));
    }
    Ok(())
}

fn run_doctor() -> Result<(), AppError> {
    let checks = [
        command_check("spectacle"),
        command_check("wl-copy"),
        command_check("notify-send"),
        command_check("curl"),
        runtime_check(),
        model_dir_check(),
    ];

    for check in &checks {
        let state = if check.ok { "ok" } else { "missing" };
        println!("[{state}] {:<12} {}", check.name, check.detail);
    }
    if checks.iter().any(|check| !check.ok) {
        return Err(AppError::MissingDependency(
            "Environment check failed; install the missing dependencies and retry".to_owned(),
        ));
    }
    println!("Environment check passed.");
    Ok(())
}

#[derive(Debug)]
struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

fn command_check(name: &'static str) -> Check {
    match find_command(name) {
        Some(path) => Check {
            name,
            ok: true,
            detail: path.display().to_string(),
        },
        None => Check {
            name,
            ok: false,
            detail: "not found in PATH".to_owned(),
        },
    }
}

fn runtime_check() -> Check {
    let output = Command::new("ldconfig").arg("-p").output();
    let found = output
        .ok()
        .map(|result| String::from_utf8_lossy(&result.stdout).contains("libonnxruntime.so"))
        .unwrap_or(false);
    Check {
        name: "onnxruntime",
        ok: found,
        detail: if found {
            "libonnxruntime.so is registered with ldconfig".to_owned()
        } else {
            "libonnxruntime.so was not found in the system library cache".to_owned()
        },
    }
}

fn model_dir_check() -> Check {
    let path = model_dir();
    Check {
        name: "model-dir",
        ok: path.is_dir(),
        detail: format!(
            "{}{}",
            path.display(),
            if path.is_dir() {
                ""
            } else {
                " (created by model setup)"
            }
        ),
    }
}

fn model_dir() -> PathBuf {
    env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local/share"))
        .join("kdeocr/models")
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn find_command(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::{Cli, CommandKind, spectacle_args, validate_png};
    use clap::Parser;

    #[test]
    fn accepts_png() {
        assert!(validate_png(b"\x89PNG\r\n\x1a\npayload").is_ok());
    }

    #[test]
    fn rejects_short_png() {
        assert!(validate_png(b"\x89PNG\r\n\x1a\n").is_err());
    }

    #[test]
    fn uses_release_capture() {
        assert!(spectacle_args().contains(&"--release-capture"));
    }

    #[test]
    fn parses_config() {
        let cli = Cli::try_parse_from(["kocr", "config"]).unwrap();
        assert!(matches!(cli.command, Some(CommandKind::Config)));
    }

    #[test]
    fn parses_daemon() {
        let cli = Cli::try_parse_from(["kocr", "daemon"]).unwrap();
        assert!(matches!(cli.command, Some(CommandKind::Daemon)));
    }

    #[test]
    fn parses_capture_ocr() {
        let cli = Cli::try_parse_from(["kocr", "capture", "-o"]).unwrap();
        let Some(CommandKind::Capture(args)) = cli.command else {
            panic!("expected capture command");
        };
        assert!(args.ocr);
    }

    #[test]
    fn parses_image() {
        let cli = Cli::try_parse_from(["kocr", "image", "image.png"]).unwrap();
        assert!(matches!(cli.command, Some(CommandKind::Image(_))));
    }
}
