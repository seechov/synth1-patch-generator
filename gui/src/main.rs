//! Synth1GAN GUI — generates Synth1 VST presets using a trained WGAN-GP model.
//!
//! Workflow:
//!   1. Point the app at a folder that contains generator.onnx + normalization.json
//!   2. Choose output folder and preset count
//!   3. Click Generate → .sy1 files appear in the output folder

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use eframe::egui::{self, RichText, ScrollArea, TextEdit};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use tract_onnx::prelude::*;

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

type TracModel = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    Graph<TypedFact, Box<dyn TypedOp>>,
>;

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

/// Write a single preset to a .sy1 file.
fn write_sy1(
    path: &Path,
    preset_name: &str,
    params: &HashMap<String, i64>,
) -> std::io::Result<()> {
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
    fs::write(path, lines.join("\n") + "\n")
}

// ─── Application state ────────────────────────────────────────────────────────

struct App {
    model_dir: String,
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
        Self {
            model_dir: String::new(),
            output_dir: String::new(),
            num_presets: 16,
            bank_name: "SynthGAN".to_string(),
            log: Vec::new(),
            model: None,
            norm_params: None,
        }
    }

    fn log(&mut self, msg: impl Into<String>) {
        self.log.push(msg.into());
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
        let mut ok = 0u32;
        let mut err = 0u32;

        {
            let model = self.model.as_ref().unwrap();
            let params = self.norm_params.as_ref().unwrap();
            let latent_dim = params.latent_dim;
            let output_dim = params.output_dim;

            new_logs.push(format!("Generating {} presets…", num_presets));

            for i in 0..num_presets {
                let output = generate_one(model, latent_dim, output_dim);
                let preset_params = decode_preset(&output, params);
                let preset_name = format!("{}-{:03}", bank, i + 1);
                let file_path = out_dir.join(format!("{:03}.sy1", i + 1));

                match write_sy1(&file_path, &preset_name, &preset_params) {
                    Ok(_) => ok += 1,
                    Err(e) => {
                        new_logs.push(format!("  ✗ {:03}.sy1: {}", i + 1, e));
                        err += 1;
                    }
                }
            }
        }

        new_logs.push(format!(
            "✓ Done: {} presets written to {}  ({} errors)",
            ok,
            out_dir.display(),
            err
        ));
        self.log.extend(new_logs);
    }

    fn model_loaded(&self) -> bool {
        self.model.is_some() && self.norm_params.is_some()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("settings").min_width(260.0).show(ctx, |ui| {
            ui.add_space(8.0);
            ui.heading("Synth1GAN");
            ui.add_space(4.0);
            ui.separator();

            // ── Model folder ──
            ui.add_space(8.0);
            ui.label(RichText::new("Model folder").strong());
            ui.horizontal(|ui| {
                ui.add(
                    TextEdit::singleline(&mut self.model_dir)
                        .hint_text("path/to/model/")
                        .desired_width(160.0),
                );
                if ui.button("…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.model_dir = p.display().to_string();
                    }
                }
            });
            let btn_label = if self.model_loaded() {
                "↺ Reload"
            } else {
                "Load model"
            };
            if ui.button(btn_label).clicked() {
                self.load_model();
            }
            if self.model_loaded() {
                ui.colored_label(egui::Color32::GREEN, "● model ready");
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
                    .add_sized([ui.available_width(), 36.0], egui::Button::new("⚡ Generate"))
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
