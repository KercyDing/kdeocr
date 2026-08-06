use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use ruzstd::decoding::StreamingDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;
use tempfile::TempDir;

use super::{
    ModelError, default_model_path, load_index, model_path, record_install, resolve_profile,
    sync_config,
};

const MAX_MODEL_WINDOW_SIZE: u64 = 256 * 1024 * 1024;

pub(crate) fn run(selector: &str, path: Option<&Path>) -> Result<(), ModelError> {
    sync_config()?;
    let index = load_index()?;
    let (name, profile) = resolve_profile(&index, selector)?;
    if model_path(name)?.join("manifest.toml").is_file() {
        return Err(ModelError::Operation(format!(
            "{name} is already installed"
        )));
    }
    let destination = match path {
        Some(path) => std::path::absolute(path).map_err(|error| {
            ModelError::Operation(format!("could not resolve model path: {error}"))
        })?,
        None => default_model_path(name),
    };
    if destination.exists() {
        return Err(ModelError::Operation(format!(
            "model destination already exists: {}",
            destination.display()
        )));
    }
    let model_root = destination
        .parent()
        .ok_or_else(|| ModelError::Operation("invalid model path".to_owned()))?;
    fs::create_dir_all(model_root).map_err(|error| {
        ModelError::Operation(format!("could not create model directory: {error}"))
    })?;
    let temporary = TempDir::new().map_err(|error| {
        ModelError::Operation(format!("could not create temporary directory: {error}"))
    })?;
    let archive_path = temporary.path().join("model.tar.zst");
    let curl = find_command("curl")
        .ok_or_else(|| ModelError::Operation("curl not found in PATH".to_owned()))?;
    let status = Command::new(curl)
        .args(["-fL", "--retry", "3", "--progress-bar", "--output"])
        .arg(&archive_path)
        .arg(&profile.url)
        .status()
        .map_err(|error| ModelError::Operation(format!("could not start curl: {error}")))?;
    if !status.success() {
        return Err(ModelError::Operation(format!("download failed: {status}")));
    }
    let actual = file_sha256(&archive_path)?;
    if actual != profile.sha256 {
        return Err(ModelError::Operation(format!(
            "SHA-256 mismatch: expected {}, got {actual}",
            profile.sha256
        )));
    }
    let extracted = tempfile::Builder::new()
        .prefix(".kdeocr-extract-")
        .tempdir_in(model_root)
        .map_err(|error| {
            ModelError::Operation(format!("could not create extraction directory: {error}"))
        })?;
    let file = File::open(&archive_path)
        .map_err(|error| ModelError::Operation(format!("could not open archive: {error}")))?;
    let decoder = StreamingDecoder::new_with_max_window_size(file, MAX_MODEL_WINDOW_SIZE)
        .map_err(|error| ModelError::Operation(format!("could not decode archive: {error}")))?;
    Archive::new(decoder)
        .unpack(extracted.path())
        .map_err(|error| ModelError::Operation(format!("could not extract archive: {error}")))?;
    let package = single_directory(extracted.path())?;
    if !package.join("manifest.toml").is_file() {
        return Err(ModelError::Operation(
            "archive does not contain a model manifest".to_owned(),
        ));
    }
    fs::rename(package, &destination).map_err(|error| {
        ModelError::Operation(format!("could not install model profile: {error}"))
    })?;
    if let Err(error) = record_install(name, &destination) {
        fs::remove_dir_all(&destination).map_err(|cleanup| {
            ModelError::Operation(format!(
                "could not save model configuration: {error}; cleanup failed: {cleanup}"
            ))
        })?;
        return Err(error);
    }
    println!("Installed {name}");
    println!("Path: {}", display_path(&destination));
    Ok(())
}

fn single_directory(root: &Path) -> Result<PathBuf, ModelError> {
    let mut entries = fs::read_dir(root)
        .map_err(|error| ModelError::Operation(format!("could not inspect archive: {error}")))?;
    let first = entries
        .next()
        .ok_or_else(|| ModelError::Operation("archive is empty".to_owned()))?
        .map_err(|error| ModelError::Operation(format!("could not inspect archive: {error}")))?
        .path();
    if !first.is_dir() || entries.next().is_some() {
        return Err(ModelError::Operation(
            "archive must contain one top-level directory".to_owned(),
        ));
    }
    Ok(first)
}

fn file_sha256(path: &Path) -> Result<String, ModelError> {
    let bytes = fs::read(path)
        .map_err(|error| ModelError::Operation(format!("could not read archive: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn find_command(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

fn display_path(path: &Path) -> String {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return path.display().to_string();
    };
    match path.strip_prefix(home) {
        Ok(relative) => format!("~/{}", relative.display()),
        Err(_) => path.display().to_string(),
    }
}
