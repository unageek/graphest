# Building Graphest from Source

## Prerequisites

### macOS

1. Command Line Tools for Xcode

   ```bash
   xcode-select --install
   ```

1. [Homebrew](https://brew.sh)

1. Autotools

   ```bash
   brew install autoconf automake libtool
   ```

1. [Node.js](https://nodejs.org/en/download/) v24 or later

1. [Rust](https://rustup.rs)

### Windows

1. [MSYS2](https://www.msys2.org)

1. [Node.js](https://nodejs.org/en/download/) v24 or later

1. Make Node.js available in MSYS2

   Open `C:\msys64\mingw64.ini` and uncomment the following line:

   ```ini
   MSYS2_PATH_TYPE=inherit
   ```

1. Install build tools

   Open Start > MSYS2 > MSYS2 MINGW64 and run the following command:

   ```bash
   pacman -S diffutils git m4 make mingw-w64-x86_64-autotools mingw-w64-x86_64-clang mingw-w64-x86_64-gcc mingw-w64-x86_64-rustup
   ```

   All commands below must be run in the MSYS2 MINGW64 terminal.

1. Select Rust toolchain

   Set `x86_64-pc-windows-gnu` as the default host:

   ```bash
   rustup set default-host x86_64-pc-windows-gnu
   ```

   See [Windows - The rustup book](https://rust-lang.github.io/rustup/installation/windows.html) for details.

### Ubuntu

1. Command line tools and libraries

   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y autoconf automake build-essential curl git libclang-dev libtool m4
   ```

   [libraries required to run Electron](https://github.com/electron/electron/issues/26673):

   ```bash
   sudo apt install -y libasound2t64 libatk-bridge2.0-0 libatk1.0-0 libgbm1 libgdk-pixbuf-2.0-0 libgtk-3-0 libnss3
   ```

1. [Node.js](https://nodejs.org/en/download/) v24 or later

1. [Rust](https://rustup.rs)

## Build

1. Install Yarn (if you don't have it yet)

   ```bash
   npm install -g yarn
   ```

1. Clone the repo and install Node.js dependencies

   ```bash
   git clone https://github.com/unageek/graphest.git
   cd graphest
   yarn
   ```

1. Run the app

   ```bash
   yarn start
   ```
