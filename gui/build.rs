#[cfg(windows)]
fn main() {
    // Embed the icon (and version info) into the Windows executable.
    let mut res = winres::WindowsResource::new();
    res.set_icon("assets/synth1gan.ico");
    res.set("ProductName", "Synth1GAN");
    res.set(
        "FileDescription",
        "Synth1 VST preset generator powered by a WGAN-GP",
    );
    res.set("LegalCopyright", "Copyright 2026 Aleksei Sychev");
    if let Err(e) = res.compile() {
        eprintln!("warning: winres failed: {}", e);
    }
}

#[cfg(not(windows))]
fn main() {}
