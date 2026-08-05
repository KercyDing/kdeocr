use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Index {
    format: u32,
    profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Deserialize)]
struct Profile {
    id: u32,
    url: String,
    sha256: String,
}

fn main() {
    println!("cargo:rerun-if-changed=index.toml");
    let source = fs::read_to_string("index.toml").expect("could not read index.toml");
    let index: Index = toml::from_str(&source).expect("could not parse index.toml");
    let mut generated =
        String::from("fn model_index() -> ModelIndex {\n    ModelIndex {\n        format: ");
    generated.push_str(&index.format.to_string());
    generated.push_str(",\n        profiles: [\n");
    for (name, profile) in index.profiles {
        generated.push_str("            (");
        push_string(&mut generated, &name);
        generated.push_str(", ");
        generated.push_str(&profile.id.to_string());
        generated.push_str(", ");
        push_string(&mut generated, &profile.url);
        generated.push_str(", ");
        push_string(&mut generated, &profile.sha256);
        generated.push_str("),\n");
    }
    generated.push_str(
        "        ]\n            .into_iter()\n            .map(|(name, id, url, sha256)| (\n                name.to_owned(),\n                ModelProfile {\n                    id,\n                    url: url.to_owned(),\n                    sha256: sha256.to_owned(),\n                },\n            ))\n            .collect(),\n    }\n}\n",
    );

    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set")).join("model_index.rs");
    fs::write(output, generated).expect("could not write generated model index");
}

fn push_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        for escaped in character.escape_default() {
            output.push(escaped);
        }
    }
    output.push('"');
}
