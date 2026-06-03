FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Remove yarn repo from base image — its GPG key is expired and breaks apt-get update
RUN rm -f /etc/apt/sources.list.d/yarn.list

# System deps for Rust GUI compilation (Linux only — GUI builds for Windows cross-compile separately)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libfontconfig1-dev \
    libxcb1-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python training dependencies (PyTorch CPU edition for the container)
# For CUDA training, run train.py on your host Windows machine with a CUDA-enabled PyTorch
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt
