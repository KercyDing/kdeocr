use super::{ModelError, load_index, model_path, resolve_profile, sync_config, write_config};

pub(crate) fn run(selector: &str) -> Result<(), ModelError> {
    sync_config()?;
    let index = load_index()?;
    let (name, _) = resolve_profile(&index, selector)?;
    if !model_path(name).join("manifest.toml").is_file() {
        return Err(ModelError::Operation(format!("{name} is not installed")));
    }
    write_config(name)?;
    println!("Using {name}");
    Ok(())
}
