#!/usr/bin/env python3
"""Generate English adjective and noun corpora for preset naming.

Fetch real English word lists (dariusk/corpora) over the network, then clean
them into single lowercase tokens. Pad to exactly 1000 unique words per corpus
with curated synth/sound-themed fallback words.

Output:
  gui/corpus/adjectives.txt  (1000 lowercase adjectives, one per line)
  gui/corpus/nouns.txt       (1000 lowercase nouns, one per line)

The words are token-safe: [a-z] only, no spaces/hyphens/apostrophes, so they
embed cleanly into preset names via `include_str!`.
"""

import json
import re
import urllib.request
from pathlib import Path

ADJS_URL = "https://raw.githubusercontent.com/dariusk/corpora/master/data/words/adjs.json"
NOUNS_URL = "https://raw.githubusercontent.com/dariusk/corpora/master/data/words/nouns.json"

# Curated fallback words (synth / sound / electronic-music themed) used to pad
# each corpus up to the target size with clean, evocative terms.
ADJ_FALLBACK = [
    "abrasive", "acidic", "airy", "ambient", "analog", "analogue", "arpeggiated",
    "atonal", "bassy", "bent", "bitcrushed", "booming", "breakbeat", "brittle",
    "chordal", "chorused", "chromatic", "clipped", "cloudy", "coarse", "colourful",
    "compressed", "connective", "crisp", "dark", "deep", "delayed", "detuned",
    "digital", "dissonant", "distorted", "dreamy", "driven", "droney", "dubbed",
    "dynamic", "ebbing", "echoing", "effervescent", "elastic", "electronic",
    "enveloped", "ethereal", "evolving", "filtered", "flanged", "floating",
    "formant", "fractal", "fuzzy", "gated", "glassy", "glitchy", "granular",
    "harmonic", "haunting", "hypnotic", "industrial", "layered", "liquid",
    "loopable", "luminous", "magnetic", "metallic", "microtonal", "modular",
    "monophonic", "morphing", "murky", "muted", "nasal", "noisy", "organic",
    "oscillating", "overdriven", "panned", "percussive", "phased", "phasing",
    "plastic", "plucked", "polyphonic", "pulsating", "punchy", "resonant",
    "reverbant", "reverberant", "ringing", "sample", "saturated", "sawtooth",
    "sequenced", "shimmering", "sidechained", "sine", "sliding", "slow", "smooth",
    "sparkling", "spatial", "square", "steppy", "stereo", "stuttering", "synth",
    "textural", "thumping", "trancey", "tremolo", "triangular", "twisted",
    "vibrant", "vintage", "voltage", "warm", "warbling", "wavy", "weightless",
]

NOUN_FALLBACK = [
    "arpeggio", "acid", "attack", "bass", "bassline", "beat", "bleep", "board",
    "break", "cabinet", "cadence", "chorus", "circuit", "clock", "coda", "combo",
    "compressor", "controller", "crossfade", "crystal", "cutoff", "decay",
    "delay", "detune", "distortion", "drone", "drop", "drum", "dub", "echo",
    "envelope", "filter", "flange", "frequency", "gate", "glide", "grain", "grid",
    "groove", "harmony", "hat", "impedance", "interval", "kick", "knob", "layer",
    "lead", "loop", "matrix", "melody", "meter", "midi", "mixer", "mode", "modem",
    "modulation", "monitor", "moog", "mosfet", "multiplexer", "noise", "oscillator",
    "overdrive", "pad", "pan", "patch", "phaser", "plate", "polyphony", "preset",
    "pulse", "reverb", "riff", "ring", "sample", "saw", "sequence", "sequencer",
    "shaper", "snare", "spectrum", "square", "step", "sub", "sustain", "swell",
    "synthesizer", "sync", "table", "tempo", "threshold", "timbre", "tone",
    "transistor", "tremolo", "triangle", "trigger", "tube", "vamp", "velocity",
    "vibrato", "voice", "volt", "wave", "waveform", "wavetable",
]

# Words to drop even if present in the source lists (proper names, non-words,
# or awkward terms that pollute synth preset names).
BLACKLIST = {
    "allies", "armour", "aristotelian", "arthurian", "barrymore", "bohemian",
    "brethren", "balls", "bases", "begun", "bones", "cabot", "catholicism",
    "caught", "chihuahua", "china", "christianity", "colors", "concur",
    "consist", "contents", "drank", "easter", "falls", "frenchman", "fries",
    "google", "greens", "hands", "heads", "hearts", "hello", "jenny", "jones",
    "joseph", "lowry", "martin", "mayer", "means", "mosaic", "oceanic",
    "orientalism", "peter", "pharaoh", "proctor", "pueblo", "pullman",
    "ralph", "rocks", "rodeo", "saturday", "sister", "snead", "sperm",
    "stocks", "stole", "syrah", "terran", "terry", "tights", "tudor",
    "tuesday", "twenties", "woodward", "words",
}

# Regex for tokens we keep: one or more alphabetic unicode letters.
TOKEN_RE = re.compile(r"^[a-z]+$")


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def clean(entries, target_key, blacklist):
    out = []
    seen = set()
    for raw in entries:
        # Lowercase and collapse any hyphen/apostrophe into a separator so we
        # can split multi-part entries into single clean tokens.
        for cand in re.split(r"[-'’]", raw.strip()):
            tok = re.sub(r"[^a-z]+", "", cand.lower())
            if not TOKEN_RE.match(tok):
                continue
            if tok in blacklist or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


def build(fallback, url, key):
    data = json.loads(fetch(url))
    words = clean(data[key], key, BLACKLIST)
    # Pad with curated fallback words until we reach 1000 unique tokens.
    for w in fallback:
        if w not in words:
            words.append(w)
    return words[:1000]


def main():
    out = Path(__file__).resolve().parent.parent / "corpus"
    out.mkdir(parents=True, exist_ok=True)

    adjectives = build(ADJ_FALLBACK, ADJS_URL, "adjs")
    nouns = build(NOUN_FALLBACK, NOUNS_URL, "nouns")

    (out / "adjectives.txt").write_text("\n".join(adjectives) + "\n", encoding="utf-8")
    (out / "nouns.txt").write_text("\n".join(nouns) + "\n", encoding="utf-8")

    print(f"adjectives: {len(adjectives)} unique -> {out / 'adjectives.txt'}")
    print(f"nouns:       {len(nouns)} unique -> {out / 'nouns.txt'}")


if __name__ == "__main__":
    main()
