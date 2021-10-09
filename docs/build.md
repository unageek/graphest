# Building Graphest from Source

## Prerequisites

### macOS

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

### Windows

1. [Chocolatey](https://chocolatey.org/install)

1. Node.js and MSYS2

   Open PowerShell as an administrator, then run:

   ```ps
   choco install msys2 nodejs
   ```

   Add the following directories to PATH:

   ```
   C:\tools\msys64\usr\bin
   C:\tools\msys64\mingw64\bin
   ```

1. Build tools

   ```ps
   pacman -S pacman-mirrors
   pacman -S diffutils git m4 make mingw-w64-x86_64-clang mingw-w64-x86_64-gcc
   ```

1. [Rust](https://rustup.rs)

   Set `x86_64-pc-windows-gnu` as the default host either on installation:

   ```ps
   .\rustup-init --default-host x86_64-pc-windows-gnu
   ```

   or after installation:

   ```ps
   rustup set default-host x86_64-pc-windows-gnu
   ```

### Ubuntu

1. Command line tools and libraries

   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y build-essential curl git libclang-dev m4 nodejs npm
   ```

   [libraries required to run Electron](https://github.com/electron/electron/issues/26673):

   ```bash
   sudo apt install -y libatk-bridge2.0-0 libatk1.0-0 libgbm1 libgconf-2-4 libgdk-pixbuf2.0-0 libgtk-3-0 libnss3
   ```

1. [Rust](https://rustup.rs)

## Build

1. Clone the repo and install Node.js dependencies

   ```bash
   git clone https://github.com/unageek/graphest.git
   cd graphest
   npm install
   ```

1. Run the app

   ```bash
   npm start
   ```
