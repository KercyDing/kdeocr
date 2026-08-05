use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::OcrError;

#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) profile: String,
    pub(crate) charset: String,
    pub(crate) charset_sha256: String,
    pub(crate) charset_count: usize,
    pub(crate) blank_index: usize,
    pub(crate) space_index: usize,
    pub(crate) det: TensorManifest,
    pub(crate) rec: TensorManifest,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TensorManifest {
    pub(crate) model: String,
    pub(crate) sha256: String,
    pub(crate) input_name: String,
    pub(crate) output_name: String,
    pub(crate) input_shape: Vec<i64>,
    pub(crate) output_shape: Vec<i64>,
}

pub(crate) fn load(model_dir: &Path) -> Result<(Manifest, Vec<char>), OcrError> {
    let manifest = read_manifest(model_dir)?;
    validate_manifest(&manifest)?;
    let charset_path = checked_relative_path(model_dir, &manifest.charset)?;
    let charset = read_charset(&charset_path, &manifest)?;
    let det_model = checked_relative_path(model_dir, &manifest.det.model)?;
    let rec_model = checked_relative_path(model_dir, &manifest.rec.model)?;
    verify_file(&det_model, &manifest.det.sha256)?;
    verify_file(&rec_model, &manifest.rec.sha256)?;
    Ok((manifest, charset))
}

fn read_manifest(model_dir: &Path) -> Result<Manifest, OcrError> {
    let path = model_dir.join("manifest.toml");
    let content = fs::read_to_string(&path).map_err(|error| {
        OcrError::Manifest(format!("could not read {}: {error}", path.display()))
    })?;
    toml::from_str(&content)
        .map_err(|error| OcrError::Manifest(format!("could not parse {}: {error}", path.display())))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), OcrError> {
    if manifest.profile.is_empty() {
        return Err(OcrError::Manifest(format!(
            "unsupported profile {}",
            manifest.profile
        )));
    }
    if manifest.det.input_name.is_empty() || manifest.det.output_name.is_empty() {
        return Err(OcrError::Manifest(
            "detection tensor names cannot be empty".to_owned(),
        ));
    }
    if manifest.det.input_shape != [-1, 3, -1, -1] || manifest.det.output_shape != [-1, 1, -1, -1] {
        return Err(OcrError::Manifest(
            "detection tensor contract is unsupported".to_owned(),
        ));
    }
    if manifest.rec.input_name.is_empty() || manifest.rec.output_name.is_empty() {
        return Err(OcrError::Manifest(
            "recognition tensor names cannot be empty".to_owned(),
        ));
    }
    if manifest.rec.input_shape != [-1, 3, 48, -1]
        || manifest.rec.output_shape != [-1, -1, (manifest.charset_count + 2) as i64]
    {
        return Err(OcrError::Manifest(
            "recognition tensor contract is unsupported".to_owned(),
        ));
    }
    Ok(())
}

fn checked_relative_path(model_dir: &Path, value: &str) -> Result<PathBuf, OcrError> {
    let relative = Path::new(value);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(OcrError::Manifest(format!("unsafe model path: {value}")));
    }
    Ok(model_dir.join(relative))
}

fn read_charset(path: &Path, manifest: &Manifest) -> Result<Vec<char>, OcrError> {
    let content = fs::read(path).map_err(|error| {
        OcrError::ModelFile(format!("could not read {}: {error}", path.display()))
    })?;
    let actual = hex_sha256(&content);
    if actual != manifest.charset_sha256 {
        return Err(OcrError::ModelFile(format!(
            "charset SHA-256 mismatch: expected {}, got {actual}",
            manifest.charset_sha256
        )));
    }
    let mut charset = Vec::with_capacity(manifest.charset_count);
    for line in String::from_utf8(content)
        .map_err(|error| OcrError::ModelFile(format!("charset is not UTF-8: {error}")))?
        .lines()
    {
        let mut chars = line.chars();
        let Some(character) = chars.next() else {
            return Err(OcrError::ModelFile(
                "charset contains an empty line".to_owned(),
            ));
        };
        if chars.next().is_some() {
            return Err(OcrError::ModelFile(format!(
                "charset entry is not one character: {line:?}"
            )));
        }
        charset.push(character);
    }
    if charset.len() != manifest.charset_count {
        return Err(OcrError::ModelFile(format!(
            "charset count mismatch: expected {}, got {}",
            manifest.charset_count,
            charset.len()
        )));
    }
    Ok(charset)
}

fn verify_file(path: &Path, expected: &str) -> Result<(), OcrError> {
    let content = fs::read(path).map_err(|error| {
        OcrError::ModelFile(format!("could not read {}: {error}", path.display()))
    })?;
    let actual = hex_sha256(&content);
    if actual != expected {
        return Err(OcrError::ModelFile(format!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(())
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
