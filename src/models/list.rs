use std::env;
use std::path::{Path, PathBuf};

use comfy_table::{Cell, Color, Table, presets::UTF8_BORDERS_ONLY};

use super::{ModelError, load_index, model_path, selected_profile, sync_config};

pub(crate) fn run() -> Result<(), ModelError> {
    sync_config()?;
    let index = load_index()?;
    let active = selected_profile().ok();
    let mut table = Table::new();
    table
        .enforce_styling()
        .load_style(UTF8_BORDERS_ONLY)
        .set_header(["ID", "Model", "Location"]);
    let mut profiles: Vec<_> = index.profiles.iter().collect();
    profiles.sort_by_key(|(_, profile)| profile.id);
    let id_width = profiles
        .iter()
        .map(|(_, profile)| profile.id.to_string().len())
        .max()
        .unwrap_or(1)
        + 1;
    for (name, profile) in profiles {
        let path = model_path(name);
        let id = if active.as_deref() == Some(name.as_str()) {
            format!("*{:>width$}", profile.id, width = id_width - 1)
        } else {
            format!("{:>id_width$}", profile.id)
        };
        if path.join("manifest.toml").is_file() {
            table.add_row([
                Cell::new(id.clone()),
                Cell::new(name).fg(Color::Green),
                Cell::new(display_path(&path)),
            ]);
        } else {
            table.add_row([Cell::new(id), Cell::new(name), Cell::new("Not installed")]);
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
