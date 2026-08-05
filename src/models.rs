mod install;
mod list;
mod uninstall;
mod r#use;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use clap::Args;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) use install::run as install;
pub(crate) use list::run as list;
pub(crate) use uninstall::run as uninstall;
pub(crate) use r#use::run as use_model;

include!(concat!(env!("OUT_DIR"), "/model_index.rs"));

const CONFIG_FILE: &str = "kdeocr/config.toml";
const DEFAULT_SHORTCUT: &str = "Alt+1";

#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct ModelConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shortcut: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default)]
    models: BTreeMap<String, InstalledModel>,
    #[serde(flatten)]
    extra: BTreeMap<String, toml::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            shortcut: Some(DEFAULT_SHORTCUT.to_owned()),
            model: None,
            models: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

impl ModelConfig {
    fn is_empty(&self) -> bool {
        self.shortcut.is_none()
            && self.model.is_none()
            && self.models.is_empty()
            && self.extra.is_empty()
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
struct InstalledModel {
    path: String,
}

#[derive(Debug)]
pub(crate) struct ModelIndex {
    pub(crate) format: u32,
    pub(crate) profiles: BTreeMap<String, ModelProfile>,
}

#[derive(Debug)]
pub(crate) struct ModelProfile {
    pub(crate) id: u32,
    pub(crate) url: String,
    pub(crate) sha256: String,
}

#[derive(Debug, Args)]
pub(crate) struct ProfileArgs {
    /// Model ID or name
    #[arg(value_name = "ID_OR_MODEL")]
    pub(crate) profile: String,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("{0}")]
    Index(String),

    #[error("{0}")]
    Operation(String),
}

pub(crate) fn load_index() -> Result<ModelIndex, ModelError> {
    let index = model_index();
    if index.format != 1 {
        return Err(ModelError::Index(format!(
            "unsupported model index format {}",
            index.format
        )));
    }
    validate_index(&index)?;
    Ok(index)
}

pub(crate) fn selected_profile() -> Result<String, ModelError> {
    sync_config()?;
    let index = load_index()?;
    let configured = read_config()?;
    let installed = installed_profiles(&index);
    if let Some(configured) = configured {
        if installed.iter().any(|name| name == &configured) {
            return Ok(configured);
        }
        if let Some(profile) = index.profiles.get(&configured)
            && let Some(name) = installed.iter().find(|name| {
                index
                    .profiles
                    .get(*name)
                    .is_some_and(|candidate| candidate.id > profile.id)
            })
        {
            return Ok(name.clone());
        }
    }
    installed
        .into_iter()
        .next()
        .ok_or_else(|| ModelError::Operation("no model profile is installed".to_owned()))
}

pub(crate) fn shortcut() -> Result<String, ModelError> {
    sync_config()?;
    read_model_config()?
        .shortcut
        .ok_or_else(|| ModelError::Operation("shortcut is missing from config".to_owned()))
}

pub(crate) fn model_path(name: &str) -> PathBuf {
    model_root().join(name)
}

pub(crate) fn resolve_profile<'a>(
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

pub(crate) fn installed_profiles(index: &ModelIndex) -> Vec<String> {
    let mut profiles: Vec<_> = index
        .profiles
        .iter()
        .filter(|(name, _)| model_path(name).join("manifest.toml").is_file())
        .map(|(name, _)| name.clone())
        .collect();
    profiles.sort_by_key(|name| index.profiles[name].id);
    profiles
}

pub(crate) fn config_path() -> PathBuf {
    env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".config")
        })
        .join(CONFIG_FILE)
}

pub(crate) fn read_config() -> Result<Option<String>, ModelError> {
    Ok(read_model_config()?.model)
}

pub(crate) fn sync_config() -> Result<(), ModelError> {
    let index = load_index()?;
    let path = config_path();
    let had_config = path.is_file();
    let mut config = read_model_config()?;
    let installed = installed_profiles(&index);
    let shortcut = config
        .shortcut
        .clone()
        .unwrap_or_else(|| DEFAULT_SHORTCUT.to_owned());
    let models = installed
        .iter()
        .map(|name| {
            (
                name.clone(),
                InstalledModel {
                    path: model_path(name).display().to_string(),
                },
            )
        })
        .collect();
    let model = config
        .model
        .as_ref()
        .filter(|name| installed.iter().any(|installed| installed == *name))
        .cloned()
        .or_else(|| installed.first().cloned());
    let changed = config.shortcut.as_deref() != Some(&shortcut)
        || config.model != model
        || config.models != models;
    config.shortcut = Some(shortcut);
    config.model = model;
    config.models = models;
    if changed && (had_config || !config.models.is_empty()) {
        save_config(&config)?;
    }
    Ok(())
}

fn read_model_config() -> Result<ModelConfig, ModelError> {
    let path = config_path();
    if !path.is_file() {
        return Ok(ModelConfig::default());
    }
    let content = fs::read_to_string(&path)
        .map_err(|error| ModelError::Operation(format!("could not read config: {error}")))?;
    toml::from_str::<ModelConfig>(&content)
        .map_err(|error| ModelError::Operation(format!("could not parse config: {error}")))
}

pub(crate) fn write_config(name: &str) -> Result<(), ModelError> {
    let index = load_index()?;
    let installed = installed_profiles(&index);
    let mut config = read_model_config()?;
    config.model = Some(name.to_owned());
    config.models = installed
        .iter()
        .map(|name| {
            (
                name.clone(),
                InstalledModel {
                    path: model_path(name).display().to_string(),
                },
            )
        })
        .collect();
    save_config(&config)
}

pub(crate) fn remove_config(name: &str) -> Result<(), ModelError> {
    let path = config_path();
    if !path.is_file() {
        return Ok(());
    }
    let mut config = read_model_config()?;
    config.models.remove(name);
    if config.model.as_deref() == Some(name) {
        config.model = None;
    }
    if config.is_empty() {
        fs::remove_file(path).map_err(|error| {
            ModelError::Operation(format!("could not clear model configuration: {error}"))
        })?;
        return Ok(());
    }
    save_config(&config)
}

fn save_config(config: &ModelConfig) -> Result<(), ModelError> {
    let path = config_path();
    let parent = path
        .parent()
        .ok_or_else(|| ModelError::Operation("invalid config path".to_owned()))?;
    fs::create_dir_all(parent).map_err(|error| {
        ModelError::Operation(format!("could not create config directory: {error}"))
    })?;
    let temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| ModelError::Operation(format!("could not create config file: {error}")))?;
    let content = toml::to_string_pretty(config)
        .map_err(|error| ModelError::Operation(format!("could not serialize config: {error}")))?;
    fs::write(temporary.path(), content)
        .map_err(|error| ModelError::Operation(format!("could not write config: {error}")))?;
    temporary
        .persist(&path)
        .map_err(|error| ModelError::Operation(format!("could not install config: {error}")))?;
    Ok(())
}

pub(crate) fn edit_config() -> Result<(), ModelError> {
    sync_config()?;
    let path = config_path();
    if !path.is_file() {
        save_config(&ModelConfig::default())?;
    }

    let editor = env::var_os("VISUAL")
        .filter(|value| !value.is_empty())
        .or_else(|| env::var_os("EDITOR").filter(|value| !value.is_empty()))
        .map(PathBuf::from)
        .or_else(|| find_command("xdg-open"))
        .ok_or_else(|| {
            ModelError::Operation("no text editor found; set VISUAL or EDITOR and retry".to_owned())
        })?;
    let status = Command::new(&editor).arg(&path).status().map_err(|error| {
        ModelError::Operation(format!(
            "could not start editor {}: {error}",
            editor.display()
        ))
    })?;
    if !status.success() {
        return Err(ModelError::Operation(format!(
            "editor {} exited with {status}",
            editor.display()
        )));
    }
    Ok(())
}

pub(crate) fn validate_index(index: &ModelIndex) -> Result<(), ModelError> {
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
    use std::collections::BTreeMap;

    use super::{InstalledModel, ModelConfig, model_index, resolve_profile};

    #[test]
    fn resolves_id() {
        let index = model_index();
        let (name, _) = resolve_profile(&index, "1").unwrap();
        assert_eq!(name, "ppocrv6-small-r1");
    }

    #[test]
    fn resolves_name() {
        let index = model_index();
        let (name, _) = resolve_profile(&index, "ppocrv6-small-r1").unwrap();
        assert_eq!(name, "ppocrv6-small-r1");
    }

    #[test]
    fn serializes_config() {
        let config = ModelConfig {
            shortcut: Some("Alt+1".to_owned()),
            model: Some("ppocrv6-small-r1".to_owned()),
            models: BTreeMap::from([(
                "ppocrv6-small-r1".to_owned(),
                InstalledModel {
                    path: "/models/ppocrv6-small-r1".to_owned(),
                },
            )]),
            extra: BTreeMap::new(),
        };
        let content = toml::to_string_pretty(&config).unwrap();

        assert_eq!(
            content,
            "shortcut = \"Alt+1\"\nmodel = \"ppocrv6-small-r1\"\n\n[models.ppocrv6-small-r1]\npath = \"/models/ppocrv6-small-r1\"\n"
        );
    }

    #[test]
    fn preserves_extra_config() {
        let config: ModelConfig =
            toml::from_str("model = \"ppocrv6-small-r1\"\n\n[ocr]\nthreshold = 0.5\n").unwrap();
        let content = toml::to_string_pretty(&config).unwrap();

        assert!(content.contains("ocr"));
        assert!(content.contains("threshold = 0.5"));
    }

    #[test]
    fn keeps_shortcut_config() {
        assert!(!ModelConfig::default().is_empty());
    }
}
