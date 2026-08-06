use std::fs;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::thread;
use std::time::Duration;

use thiserror::Error;
use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::OwnedObjectPath;

const BUS_NAME: &str = "io.github.KercyDing.kocr";
const SERVICE: &str = "org.kde.kglobalaccel";
const SERVICE_PATH: &str = "/kglobalaccel";
const COMPONENT: &str = "kocr";
const COPY_ACTION: &str = "copy";
const OCR_ACTION: &str = "capture-ocr";
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const SET_PRESENT_NO_AUTOLOAD: u32 = 2 | 4;

const QT_SHIFT: i32 = 0x0200_0000;
const QT_CONTROL: i32 = 0x0400_0000;
const QT_ALT: i32 = 0x0800_0000;
const QT_META: i32 = 0x1000_0000;

#[derive(Debug, Error)]
pub(crate) enum KeyboardError {
    #[error("{0}")]
    Operation(String),
}

enum Event {
    Trigger(Trigger),
    Reregister,
}

#[derive(Clone, Copy)]
enum Trigger {
    Copy,
    Ocr,
}

#[derive(Clone, PartialEq, Eq)]
struct Binding {
    name: String,
    key: i32,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Bindings {
    copy: Option<Binding>,
    ocr: Option<Binding>,
}

pub(crate) fn run<F, E>(mut on_trigger: F) -> Result<(), KeyboardError>
where
    F: FnMut(bool) -> Result<(), E>,
    E: std::fmt::Display,
{
    let connection = Connection::session()
        .map_err(|error| operation("could not connect to the session bus", error))?;
    connection
        .request_name(BUS_NAME)
        .map_err(|error| operation("another daemon is already running", error))?;

    let mut bindings = load_bindings()?;
    register(&connection, &bindings)?;
    if crate::models::selected_profile().is_err() {
        notify_no_model();
    }

    let (events_tx, events_rx) = mpsc::sync_channel(8);
    let trigger_pending = Arc::new(AtomicBool::new(false));
    spawn_listener(connection.clone(), events_tx, Arc::clone(&trigger_pending));

    let mut config_content = read_config_content();
    eprintln!("Shortcut daemon started");
    loop {
        match events_rx.recv_timeout(POLL_INTERVAL) {
            Ok(Event::Trigger(trigger)) => {
                if crate::models::selected_profile().is_err() {
                    notify_no_model();
                } else if let Err(error) = on_trigger(matches!(trigger, Trigger::Ocr)) {
                    eprintln!("\x1b[31mShortcut capture failed: {error}\x1b[0m");
                }
                trigger_pending.store(false, Ordering::Release);
            }
            Ok(Event::Reregister) => {
                if let Err(error) = register(&connection, &bindings) {
                    eprintln!("\x1b[31mShortcut registration failed: {error}\x1b[0m");
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(KeyboardError::Operation(
                    "shortcut listener stopped unexpectedly".to_owned(),
                ));
            }
        }
        reload_config(&connection, &mut config_content, &mut bindings);
    }
}

fn reload_config(
    connection: &Connection,
    previous_content: &mut Option<String>,
    current_bindings: &mut Bindings,
) {
    let content = read_config_content();
    if content == *previous_content {
        return;
    }
    *previous_content = content;

    let bindings = match load_bindings() {
        Ok(bindings) => bindings,
        Err(error) => {
            eprintln!("\x1b[31mShortcut reload failed: {error}\x1b[0m");
            return;
        }
    };
    if bindings == *current_bindings {
        return;
    }
    match register(connection, &bindings) {
        Ok(()) => {
            *current_bindings = bindings;
            eprintln!("Shortcuts reloaded");
        }
        Err(error) => {
            eprintln!("\x1b[31mShortcut reload failed: {error}\x1b[0m");
            let _ = register(connection, current_bindings);
        }
    }
}

fn spawn_listener(
    connection: Connection,
    events: SyncSender<Event>,
    trigger_pending: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        loop {
            if let Err(error) = listen(&connection, &events, &trigger_pending) {
                eprintln!("\x1b[31mShortcut listener failed: {error}\x1b[0m");
            }
            thread::sleep(POLL_INTERVAL);
            let _ = events.try_send(Event::Reregister);
        }
    });
}

fn listen(
    connection: &Connection,
    events: &SyncSender<Event>,
    trigger_pending: &AtomicBool,
) -> Result<(), KeyboardError> {
    let component = component_proxy(connection)?;
    let signals = component
        .receive_signal("globalShortcutPressed")
        .map_err(|error| operation("could not subscribe to shortcut events", error))?;
    for message in signals {
        let body = message.body();
        let Ok((component, action, _timestamp)) = body.deserialize::<(String, String, i64)>()
        else {
            continue;
        };
        let trigger = match action.as_str() {
            COPY_ACTION => Trigger::Copy,
            OCR_ACTION => Trigger::Ocr,
            _ => continue,
        };
        if component == COMPONENT
            && !trigger_pending.swap(true, Ordering::AcqRel)
            && events.try_send(Event::Trigger(trigger)).is_err()
        {
            trigger_pending.store(false, Ordering::Release);
        }
    }
    Ok(())
}

fn register(connection: &Connection, bindings: &Bindings) -> Result<(), KeyboardError> {
    let proxy = service_proxy(connection)?;
    register_action(
        &proxy,
        COPY_ACTION,
        "Capture and copy screenshot",
        bindings.copy.as_ref(),
    )?;
    register_action(
        &proxy,
        OCR_ACTION,
        "Capture screenshot and recognize text",
        bindings.ocr.as_ref(),
    )
}

fn register_action(
    proxy: &Proxy<'static>,
    action_name: &str,
    description: &str,
    binding: Option<&Binding>,
) -> Result<(), KeyboardError> {
    let action = action_id(action_name, description);
    let _: () = proxy
        .call("doRegister", &(action.clone(),))
        .map_err(|error| operation("could not register shortcut action", error))?;
    let keys = binding
        .map(|binding| vec![(vec![binding.key, 0, 0, 0],)])
        .unwrap_or_default();
    let assigned: Vec<(Vec<i32>,)> = proxy
        .call("setShortcutKeys", &(action, keys, SET_PRESENT_NO_AUTOLOAD))
        .map_err(|error| operation("could not set global shortcut", error))?;
    if let Some(binding) = binding
        && !assigned.iter().any(|(keys,)| keys.contains(&binding.key))
    {
        return Err(KeyboardError::Operation(format!(
            "shortcut is unavailable: {}",
            binding.name
        )));
    }
    Ok(())
}

fn service_proxy(connection: &Connection) -> Result<Proxy<'static>, KeyboardError> {
    Proxy::new(connection, SERVICE, SERVICE_PATH, "org.kde.KGlobalAccel")
        .map_err(|error| operation("could not connect to KGlobalAccel", error))
}

fn component_proxy(connection: &Connection) -> Result<Proxy<'static>, KeyboardError> {
    let path: OwnedObjectPath = service_proxy(connection)?
        .call("getComponent", &(COMPONENT,))
        .map_err(|error| operation("could not find shortcut component", error))?;
    Proxy::new(connection, SERVICE, path, "org.kde.kglobalaccel.Component")
        .map_err(|error| operation("could not connect to shortcut component", error))
}

fn action_id(action: &str, description: &str) -> Vec<String> {
    [COMPONENT, action, "KOCR", description]
        .map(str::to_owned)
        .to_vec()
}

fn load_bindings() -> Result<Bindings, KeyboardError> {
    let shortcuts = crate::models::shortcuts().map_err(|error| {
        KeyboardError::Operation(format!("could not load shortcut configuration: {error}"))
    })?;
    let copy = parse_binding(shortcuts.copy)?;
    let ocr = parse_binding(shortcuts.ocr)?;
    Ok(Bindings { copy, ocr })
}

fn parse_binding(shortcut: Option<String>) -> Result<Option<Binding>, KeyboardError> {
    shortcut
        .map(|name| {
            let key = parse_shortcut(&name)?;
            Ok(Binding { name, key })
        })
        .transpose()
}

fn read_config_content() -> Option<String> {
    fs::read_to_string(crate::models::config_path()).ok()
}

fn notify_no_model() {
    const TITLE: &str = "KOCR";
    const BODY: &str = "No OCR model available. Install one with kocr install 1.";

    eprintln!("\x1b[31m{BODY}\x1b[0m");
    match Command::new("notify-send")
        .args(["--app-name", TITLE, TITLE, BODY])
        .status()
    {
        Ok(status) if status.success() => {}
        Ok(status) => eprintln!("\x1b[31mCould not show notification: {status}\x1b[0m"),
        Err(error) => eprintln!("\x1b[31mCould not show notification: {error}\x1b[0m"),
    }
}

fn parse_shortcut(shortcut: &str) -> Result<i32, KeyboardError> {
    let mut parts = shortcut.split('+').peekable();
    let mut modifiers = 0;
    let mut key = None;
    while let Some(part) = parts.next() {
        let part = part.trim();
        if part.is_empty() {
            return Err(invalid_shortcut(shortcut));
        }
        let modifier = match part {
            "Mod" | "Super" | "Meta" => Some(QT_META),
            "Ctrl" | "Control" => Some(QT_CONTROL),
            "Alt" => Some(QT_ALT),
            "Shift" => Some(QT_SHIFT),
            _ => None,
        };
        if let Some(modifier) = modifier {
            if key.is_some() || modifiers & modifier != 0 {
                return Err(invalid_shortcut(shortcut));
            }
            modifiers |= modifier;
            continue;
        }
        if parts.peek().is_some() || key.replace(parse_key(part)?).is_some() {
            return Err(invalid_shortcut(shortcut));
        }
    }
    key.map(|key| modifiers | key)
        .ok_or_else(|| invalid_shortcut(shortcut))
}

fn parse_key(name: &str) -> Result<i32, KeyboardError> {
    if name.len() == 1 {
        let character = name.chars().next().expect("one-byte key has one character");
        if character.is_ascii_alphanumeric() {
            return Ok(character.to_ascii_uppercase() as i32);
        }
    }
    if let Some(number) = name
        .strip_prefix('F')
        .and_then(|value| value.parse::<i32>().ok())
        && (1..=35).contains(&number)
    {
        return Ok(0x0100_0030 + number - 1);
    }
    let key = match name {
        "Escape" => 0x0100_0000,
        "Tab" => 0x0100_0001,
        "BackSpace" | "Backspace" => 0x0100_0003,
        "Return" => 0x0100_0004,
        "Enter" => 0x0100_0005,
        "Insert" => 0x0100_0006,
        "Delete" => 0x0100_0007,
        "Pause" => 0x0100_0008,
        "Print" => 0x0100_0009,
        "Home" => 0x0100_0010,
        "End" => 0x0100_0011,
        "Left" => 0x0100_0012,
        "Up" => 0x0100_0013,
        "Right" => 0x0100_0014,
        "Down" => 0x0100_0015,
        "Page_Up" | "PageUp" => 0x0100_0016,
        "Page_Down" | "PageDown" => 0x0100_0017,
        "Space" => 0x20,
        "Apostrophe" => '\'' as i32,
        "Comma" => ',' as i32,
        "Minus" => '-' as i32,
        "Period" => '.' as i32,
        "Slash" => '/' as i32,
        "Semicolon" => ';' as i32,
        "Equal" => '=' as i32,
        "BracketLeft" => '[' as i32,
        "Backslash" => '\\' as i32,
        "BracketRight" => ']' as i32,
        "Grave" => '`' as i32,
        "XF86AudioLowerVolume" => 0x0100_0070,
        "XF86AudioMute" => 0x0100_0071,
        "XF86AudioRaiseVolume" => 0x0100_0072,
        "XF86AudioPlay" => 0x0100_0080,
        "XF86AudioStop" => 0x0100_0081,
        "XF86AudioPrev" => 0x0100_0082,
        "XF86AudioNext" => 0x0100_0083,
        "XF86AudioPause" => 0x0100_0085,
        "XF86MonBrightnessUp" => 0x0100_00b2,
        "XF86MonBrightnessDown" => 0x0100_00b3,
        _ => {
            return Err(KeyboardError::Operation(format!(
                "unsupported shortcut key: {name}"
            )));
        }
    };
    Ok(key)
}

fn invalid_shortcut(shortcut: &str) -> KeyboardError {
    KeyboardError::Operation(format!("invalid shortcut: {shortcut}"))
}

fn operation(context: &str, error: impl std::fmt::Display) -> KeyboardError {
    KeyboardError::Operation(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{QT_ALT, QT_META, QT_SHIFT, parse_binding, parse_shortcut};

    #[test]
    fn parses_shortcut() {
        assert_eq!(parse_shortcut("Alt+1").unwrap(), QT_ALT | '1' as i32);
        assert_eq!(
            parse_shortcut("Mod+Shift+Slash").unwrap(),
            QT_META | QT_SHIFT | '/' as i32
        );
        assert!(parse_shortcut("Alt+Alt+1").is_err());
    }

    #[test]
    fn accepts_disabled_binding() {
        assert!(parse_binding(None).unwrap().is_none());
    }
}
