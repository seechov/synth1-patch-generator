//! Synth1GAN GUI — generates Synth1 VST presets using a trained WGAN-GP model.
//!
//! Workflow:
//!   1. Point the app at a folder that contains generator.onnx + normalization.json
//!   2. Choose output folder and preset count
//!  3. Click Generate → a <bank-name>.zip archive appears in the output folder

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use eframe::egui::{self, RichText, ScrollArea, TextEdit};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use tract_onnx::prelude::*;

// ─── Preset-name word corpora ───────────────────────────────────────────────
// Embed the English word lists directly into the binary at compile time.
// Each corpus holds 1000 unique lowercase tokens (adjective / noun).
static ADJECTIVES: &str = include_str!("../corpus/adjectives.txt");
static NOUNS: &str = include_str!("../corpus/nouns.txt");

// ─── Normalization types (mirrors normalization.json) ─────────────────────────

#[derive(Deserialize, Clone)]
struct ContinuousStat {
    col_idx: usize,
    min: f32,
    max: f32,
    param_id: String,
}

#[derive(Deserialize, Clone)]
struct CategoricalEncoding {
    start_idx: usize,
    num_classes: usize,
    categories: Vec<i64>,
    param_id: String,
}

#[derive(Deserialize, Clone)]
struct NormParams {
    latent_dim: usize,
    output_dim: usize,
    #[allow(dead_code)]
    column_names: Vec<String>,
    continuous_stats: HashMap<String, ContinuousStat>,
    categorical_encodings: HashMap<String, CategoricalEncoding>,
}

// ─── ONNX model wrapper ───────────────────────────────────────────────────────

type TracModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

fn load_onnx(path: &Path, latent_dim: usize) -> TractResult<TracModel> {
    tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([1usize, latent_dim]).into())?
        .into_optimized()?
        .into_runnable()
}

fn generate_one(model: &TracModel, latent_dim: usize, _output_dim: usize) -> Vec<f32> {
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut rng = thread_rng();
    let noise: Vec<f32> = (0..latent_dim).map(|_| normal.sample(&mut rng)).collect();

    let tensor = tract_ndarray::Array2::from_shape_vec((1, latent_dim), noise)
        .expect("noise shape")
        .into_tensor();

    let result = model.run(tvec![tensor.into()]).expect("inference failed");
    result[0]
        .to_array_view::<f32>()
        .expect("extract output")
        .as_slice()
        .expect("contiguous output")
        .to_vec()
}

// ─── Post-processing ──────────────────────────────────────────────────────────

/// Decode one output vector into a map of param_id → integer value.
fn decode_preset(output: &[f32], params: &NormParams) -> HashMap<String, i64> {
    let mut result: HashMap<String, i64> = HashMap::new();

    // Continuous variables: denormalize from [-1, 1] → [min, max]
    for stat in params.continuous_stats.values() {
        let raw = output[stat.col_idx];
        let val = ((raw + 1.0) * (stat.max - stat.min) / 2.0 + stat.min)
            .round()
            .clamp(stat.min, stat.max) as i64;
        result.insert(stat.param_id.clone(), val);
    }

    // Categorical variables: argmax over one-hot slice → category value
    for enc in params.categorical_encodings.values() {
        let slice = &output[enc.start_idx..enc.start_idx + enc.num_classes];
        let argmax = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let value = enc.categories.get(argmax).copied().unwrap_or(0);
        result.insert(enc.param_id.clone(), value);
    }

    result
}

/// Render a preset to the .sy1 text format.
fn sy1_content(preset_name: &str, params: &HashMap<String, i64>) -> String {
    let mut lines = vec![
        preset_name.to_string(),
        "color=red".to_string(),
        "ver=106".to_string(),
    ];
    // Sort by numeric param ID for a tidy file
    let mut sorted: Vec<(&String, &i64)> = params.iter().collect();
    sorted.sort_by_key(|(k, _)| k.parse::<u32>().unwrap_or(9999));
    for (id, val) in sorted {
        lines.push(format!("{},{}", id, val));
    }
    lines.join("\n") + "\n"
}

/// Build a random preset name: a three-digit index, an adjective and a noun.
/// Example: `017 molten-horizon`.
fn preset_name(index: u32, rng: &mut impl rand::Rng) -> String {
    let adjectives: Vec<&str> = ADJECTIVES.lines().collect();
    let nouns: Vec<&str> = NOUNS.lines().collect();
    let adj = adjectives.choose(rng).copied().unwrap_or("sonic");
    let noun = nouns.choose(rng).copied().unwrap_or("preset");
    format!("{:03} {}-{}", index, adj, noun)
}

/// Return a `<base>.zip` path that does not exist yet by appending a numeric
/// suffix (`-2`, `-3`, …) when the plain name is already taken.
fn unique_zip_path(dir: &Path, base: &str) -> PathBuf {
    let candidate = dir.join(format!("{}.zip", base));
    if !candidate.exists() {
        return candidate;
    }
    let mut n = 2u32;
    loop {
        let candidate = dir.join(format!("{}-{}.zip", base, n));
        if !candidate.exists() {
            return candidate;
        }
        n += 1;
    }
}

// ─── Application state ────────────────────────────────────────────────────────

struct App {
    model_dir: String, // currently selected model folder
    output_dir: String,
    num_presets: u32,
    bank_name: String,

    log: Vec<String>,

    // Loaded at runtime
    model: Option<TracModel>,
    norm_params: Option<NormParams>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Locate the bundled model (shipped next to the executable by the installer).
        let model_dir = App::find_default_model_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        let mut app = Self {
            model_dir,
            output_dir: App::default_output_dir().unwrap_or_default(),
            num_presets: 16,
            bank_name: "SynthGAN".to_string(),
            log: Vec::new(),
            model: None,
            norm_params: None,
        };

        // Auto-load the bundled model so the user only has to click Generate.
        if !app.model_dir.is_empty() {
            app.log("Bundled model detected — loading…");
            app.load_model();
        }

        app
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.log.push(msg.into());
    }

    /// Return a sensible default output folder (the user's Documents/Synth1GAN).
    fn default_output_dir() -> Option<String> {
        if let Some(docs) = dirs_document_dir() {
            let dir = docs.join("Synth1GAN").join("presets");
            return Some(dir.display().to_string());
        }
        None
    }

    /// Try to locate the bundled model folder shipped next to the executable.
    ///
    /// Search order:
    ///   1. `<exe_dir>/model`   (Windows installer layout)
    ///   2. `<exe_dir>`         (dev convenience — files sit next to the binary)
    ///   3. `/usr/share/synth1gan/model` (Linux deb/rpm layout)
    fn find_default_model_dir() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(exe_dir) = current_exe_dir() {
            candidates.push(exe_dir.join("model"));
            candidates.push(exe_dir);
        }
        #[cfg(not(target_os = "windows"))]
        candidates.push(PathBuf::from("/usr/share/synth1gan/model"));

        for candidate in candidates {
            if candidate.join("generator.onnx").exists()
                && candidate.join("normalization.json").exists()
            {
                return Some(candidate);
            }
        }
        None
    }

    fn load_model(&mut self) {
        let dir = PathBuf::from(&self.model_dir);
        let onnx_path = dir.join("generator.onnx");
        let norm_path = dir.join("normalization.json");

        if !onnx_path.exists() {
            self.log(format!("✗ Not found: {}", onnx_path.display()));
            return;
        }
        if !norm_path.exists() {
            self.log(format!("✗ Not found: {}", norm_path.display()));
            return;
        }

        // Load normalization params first (needed to know latent_dim)
        match fs::read_to_string(&norm_path)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<NormParams>(&s).map_err(|e| e.to_string()))
        {
            Ok(params) => {
                let latent_dim = params.latent_dim;
                self.log(format!(
                    "✓ Loaded normalization.json  (latent_dim={}, output_dim={})",
                    params.latent_dim, params.output_dim
                ));
                self.norm_params = Some(params);

                // Load ONNX model
                match load_onnx(&onnx_path, latent_dim) {
                    Ok(m) => {
                        self.model = Some(m);
                        self.log("✓ Loaded generator.onnx");
                    }
                    Err(e) => {
                        self.log(format!("✗ ONNX load error: {}", e));
                        self.norm_params = None;
                    }
                }
            }
            Err(e) => {
                self.log(format!("✗ normalization.json error: {}", e));
            }
        }
    }

    fn generate(&mut self) {
        if self.model.is_none() || self.norm_params.is_none() {
            self.log("✗ Load a model first.");
            return;
        }
        if self.output_dir.is_empty() {
            self.log("✗ Choose an output folder first.");
            return;
        }

        let out_dir = PathBuf::from(&self.output_dir);
        if let Err(e) = fs::create_dir_all(&out_dir) {
            self.log(format!("✗ Cannot create output dir: {}", e));
            return;
        }

        let num_presets = self.num_presets;
        let bank = self.bank_name.clone();
        let mut new_logs: Vec<String> = Vec::new();

        let zip_path = unique_zip_path(&out_dir, &bank);

        let result = {
            let model = self.model.as_ref().unwrap();
            let params = self.norm_params.as_ref().unwrap();
            let latent_dim = params.latent_dim;
            let output_dim = params.output_dim;

            Self::write_zip_archive(
                &zip_path,
                num_presets,
                model,
                params,
                latent_dim,
                output_dim,
            )
        };

        match result {
            Ok(written) => {
                new_logs.push(format!(
                    "✓ Done: {} presets packed into {}",
                    written,
                    zip_path.display()
                ));
            }
            Err(e) => {
                new_logs.push(format!("✗ Failed to write {}: {}", zip_path.display(), e));
            }
        }

        self.log.extend(new_logs);
    }

    fn write_zip_archive(
        zip_path: &Path,
        num_presets: u32,
        model: &TracModel,
        params: &NormParams,
        latent_dim: usize,
        output_dim: usize,
    ) -> Result<usize, String> {
        let file = fs::File::create(zip_path).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipWriter::new(file);
        let options: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        let mut written = 0usize;
        let mut rng = thread_rng();
        for i in 0..num_presets {
            let output = generate_one(model, latent_dim, output_dim);
            let preset_params = decode_preset(&output, params);
            let preset_name = preset_name(i + 1, &mut rng);
            let content = sy1_content(&preset_name, &preset_params);
            let entry_name = format!("{:03}.sy1", i + 1);

            zip.start_file(entry_name, options)
                .map_err(|e| e.to_string())?;
            zip.write_all(content.as_bytes())
                .map_err(|e| e.to_string())?;
            written += 1;
        }

        zip.finish().map_err(|e| e.to_string())?;
        Ok(written)
    }

    fn model_loaded(&self) -> bool {
        self.model.is_some() && self.norm_params.is_some()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("settings")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Synth1GAN");
                ui.add_space(4.0);
                ui.separator();

                // ── Model folder ──
                ui.add_space(8.0);
                ui.label(RichText::new("Model folder").strong());
                let choose_label = if self.model_loaded() {
                    "↺ Choose model folder"
                } else {
                    "Choose model folder…"
                };
                if ui
                    .add_sized(
                        [ui.available_width(), 32.0],
                        egui::Button::new(choose_label),
                    )
                    .clicked()
                {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.model_dir = p.display().to_string();
                        self.load_model();
                    }
                }
                if self.model_loaded() {
                    ui.colored_label(egui::Color32::GREEN, "● model ready");
                } else if !self.model_dir.is_empty() {
                    ui.add(
                        egui::Label::new(
                            RichText::new(format!("Selected: {}", self.model_dir))
                                .small()
                                .color(egui::Color32::GRAY),
                        )
                        .truncate(),
                    );
                }

                ui.add_space(12.0);
                ui.separator();

                // ── Output settings ──
                ui.add_space(8.0);
                ui.label(RichText::new("Output folder").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        TextEdit::singleline(&mut self.output_dir)
                            .hint_text("path/to/output/")
                            .desired_width(160.0),
                    );
                    if ui.button("…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.output_dir = p.display().to_string();
                        }
                    }
                });

                ui.add_space(8.0);
                ui.label(RichText::new("Bank name").strong());
                ui.text_edit_singleline(&mut self.bank_name);

                ui.add_space(8.0);
                ui.label(RichText::new("Number of presets").strong());
                ui.add(egui::Slider::new(&mut self.num_presets, 1..=128).suffix(" presets"));

                ui.add_space(16.0);
                ui.separator();
                ui.add_space(8.0);

                // ── Generate button ──
                let generate_enabled = self.model_loaded() && !self.output_dir.is_empty();
                ui.add_enabled_ui(generate_enabled, |ui| {
                    if ui
                        .add_sized(
                            [ui.available_width(), 36.0],
                            egui::Button::new("⚡ Generate"),
                        )
                        .clicked()
                    {
                        self.generate();
                    }
                });

                if !generate_enabled {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Load a model and choose an output folder first.")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                }

                ui.add_space(8.0);
                if ui.small_button("Clear log").clicked() {
                    self.log.clear();
                }
            });

        // ── Log panel ──
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Log");
            ui.separator();
            ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(line);
                    }
                    if self.log.is_empty() {
                        ui.label(
                            RichText::new("Load a model to get started.")
                                .color(egui::Color32::GRAY),
                        );
                    }
                });
        });
    }
}

// ─── Path helpers ────────────────────────────────────────────────────────────

/// Directory containing the running executable.
fn current_exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(|p| p.to_path_buf())
}

/// Resolve the user's Documents directory in a cross-platform way.
///
/// On Windows, `%USERPROFILE%\Documents`. On Unix, `~/Documents`
/// (XDG documents dir is not reliably set, so we fall back to the convention).
fn dirs_document_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("USERPROFILE").map(|v| PathBuf::from(v).join("Documents"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME").map(|v| PathBuf::from(v).join("Documents"))
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Synth1GAN — Preset Generator")
            .with_inner_size([800.0, 480.0])
            .with_min_inner_size([600.0, 360.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Synth1GAN",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
