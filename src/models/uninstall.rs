use std::fs;

use super::{
    ModelError, installed_profiles, load_index, model_path, remove_config, resolve_profile,
    selected_profile, sync_config, write_config,
};

pub(crate) fn run(selector: &str) -> Result<(), ModelError> {
    sync_config()?;
    let index = load_index()?;
    let (name, profile) = resolve_profile(&index, selector)?;
    let was_active = selected_profile()? == name;
    let path = model_path(name)?;
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
    remove_config(name)?;
    if was_active {
        let installed = installed_profiles(&index)?;
        let next = installed
            .iter()
            .find(|candidate| index.profiles[*candidate].id > profile.id)
            .or_else(|| installed.first());
        if let Some(next) = next {
            write_config(next)?;
        }
    }
    println!("Uninstalled {name}");
    Ok(())
}
