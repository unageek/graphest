# Graphest

[![build](https://img.shields.io/github/workflow/status/unageek/graphest/build/master)](https://github.com/unageek/graphest/actions?query=branch%3Amaster+workflow%3Abuild)

<p align="center">
  <img src="docs/cover.png">
</p>

## Getting Started

A prebuilt version of the app is available on the [releases](https://github.com/unageek/graphest/releases) page.

## Build from Source

### Prerequisites

#### macOS

1. Command Line Tools for Xcode

   ```bash
   xcode-select --install
   ```

1. [Rust](https://rustup.rs)

1. [Homebrew](https://brew.sh)

1. Node.js and npm

   ```bash
   brew install node
   ```

#### Windows

1. [Ubuntu on WSL](https://www.microsoft.com/store/productId/9NBLGGH4MSV6)

1. A X11 server such as [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [Xming](https://sourceforge.net/projects/xming/)

1. Everything in the [Ubuntu](#ubuntu) section on WSL

#### Ubuntu

1. [Rust](https://rustup.rs)

1. Node.js and npm

   ```bash
   sudo apt install -y nodejs npm
   ```

### Build

1. Clone the repo and install Node.js dependencies

   ```bash
   git clone https://github.com/unageek/graphest.git
   cd graphest
   npm install
   ```

1. Debug the app

   ```bash
   npm start
   ```
