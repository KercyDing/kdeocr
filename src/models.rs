use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Subcommand;
use comfy_table::{Cell, Color, Table, presets::UTF8_BORDERS_ONLY};
use ruzstd::decoding::StreamingDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tar::Archive;
use tempfile::TempDir;
use thiserror::Error;

const MODEL_INDEX: &str = include_str!("../index.toml");
const MAX_MODEL_WINDOW_SIZE: u64 = 256 * 1024 * 1024;

#[derive(Debug, Subcommand)]
pub enum ModelCommand {
    /// List available models
    List,

    /// Install a model by ID or name
    Install {
        /// Model ID or name
        #[arg(value_name = "ID_OR_MODEL")]
        profile: String,
    },

    /// Uninstall a model by ID or name
    Uninstall {
        /// Model ID or name
        #[arg(value_name = "ID_OR_MODEL")]
        profile: String,
    },
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("{0}")]
    Index(String),

    #[error("{0}")]
    Operation(String),
}

#[derive(Debug, Deserialize)]
struct ModelIndex {
    format: u32,
    profiles: BTreeMap<String, ModelProfile>,
}

#[derive(Debug, Deserialize)]
struct ModelProfile {
    id: u32,
    url: String,
    sha256: String,
}

pub fn run(command: ModelCommand) -> Result<(), ModelError> {
    let index: ModelIndex =
        toml::from_str(MODEL_INDEX).map_err(|error| ModelError::Index(error.to_string()))?;
    if index.format != 1 {
        return Err(ModelError::Index(format!(
            "unsupported model index format {}",
            index.format
        )));
    }
    validate_index(&index)?;

    match command {
        ModelCommand::List => list(&index),
        ModelCommand::Install { profile } => install(&index, &profile),
        ModelCommand::Uninstall { profile } => uninstall(&index, &profile),
    }
}

fn uninstall(index: &ModelIndex, selector: &str) -> Result<(), ModelError> {
    let (name, _) = resolve_profile(index, selector)?;
    let path = model_root().join(name);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        ModelError::Operation(format!("model profile is not installed: {error}"))
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(ModelError::Operation(format!(
            "refusing to remove invalid model directory: {}",
            path.display()
        )));
    }
    if !path.join("manifest.toml").is_file() {
        return Err(ModelError::Operation(format!(
            "refusing to remove directory without a model manifest: {}",
            path.display()
        )));
    }

    fs::remove_dir_all(&path).map_err(|error| {
        ModelError::Operation(format!("could not uninstall model profile: {error}"))
    })?;
    println!("Uninstalled {name}");
    Ok(())
}

fn list(index: &ModelIndex) -> Result<(), ModelError> {
    let root = model_root();
    let mut table = Table::new();
    table
        .enforce_styling()
        .load_style(UTF8_BORDERS_ONLY)
        .set_header(["ID", "Model", "Location"]);
    let mut profiles: Vec<_> = index.profiles.iter().collect();
    profiles.sort_by_key(|(_, profile)| profile.id);
    for (name, profile) in profiles {
        let path = root.join(name);
        if path.join("manifest.toml").is_file() {
            table.add_row([
                Cell::new(profile.id),
                Cell::new(name).fg(Color::Green),
                Cell::new(display_path(&path)),
            ]);
        } else {
            table.add_row([
                Cell::new(profile.id),
                Cell::new(name),
                Cell::new("Not installed"),
            ]);
        }
    }
    println!("{table}");
    Ok(())
}

fn display_path(path: &Path) -> String {
    let Some(home) = env::var_os("HOME").map(PathBuf::from) else {
        return path.display().to_string();
    };
    match path.strip_prefix(home) {
        Ok(relative) => format!("~/{}", relative.display()),
        Err(_) => path.display().to_string(),
    }
}

fn install(index: &ModelIndex, selector: &str) -> Result<(), ModelError> {
    let (name, profile) = resolve_profile(index, selector)?;
    let destination = model_root().join(name);
    if destination.exists() {
        return Err(ModelError::Operation(format!(
            "{name} is already installed"
        )));
    }

    let model_root = model_root();
    fs::create_dir_all(&model_root).map_err(|error| {
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
        .tempdir_in(&model_root)
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
    println!("Installed {name}");
    println!("Path: {}", display_path(&destination));
    Ok(())
}

fn validate_index(index: &ModelIndex) -> Result<(), ModelError> {
    let mut ids = BTreeSet::new();
    for (name, profile) in &index.profiles {
        if profile.id == 0 {
            return Err(ModelError::Index(format!(
                "model ID must be positive: {name}"
            )));
        }
        if !ids.insert(profile.id) {
            return Err(ModelError::Index(format!(
                "duplicate model ID: {}",
                profile.id
            )));
        }
    }
    Ok(())
}

fn resolve_profile<'a>(
    index: &'a ModelIndex,
    selector: &str,
) -> Result<(&'a str, &'a ModelProfile), ModelError> {
    if let Ok(id) = selector.parse::<u32>() {
        return index
            .profiles
            .iter()
            .find(|(_, profile)| profile.id == id)
            .map(|(name, profile)| (name.as_str(), profile))
            .ok_or_else(|| ModelError::Index(format!("unknown model ID: {id}")));
    }
    index
        .profiles
        .get_key_value(selector)
        .map(|(name, profile)| (name.as_str(), profile))
        .ok_or_else(|| ModelError::Index(format!("unknown model profile: {selector}")))
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

fn model_root() -> PathBuf {
    env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local/share")
        })
        .join("kdeocr/models")
}

fn find_command(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::{MODEL_INDEX, ModelIndex, resolve_profile};

    #[test]
    fn resolves_id() {
        let index: ModelIndex = toml::from_str(MODEL_INDEX).unwrap();
        let (name, _) = resolve_profile(&index, "1").unwrap();
        assert_eq!(name, "ppocrv6-small-r1");
    }

    #[test]
    fn resolves_name() {
        let index: ModelIndex = toml::from_str(MODEL_INDEX).unwrap();
        let (name, _) = resolve_profile(&index, "ppocrv6-small-r1").unwrap();
        assert_eq!(name, "ppocrv6-small-r1");
    }
}
