# Building Graphest from Source

## Prerequisites

### macOS

1. Command Line Tools for Xcode

   ```bash
   xcode-select --install
   ```

1. [Homebrew](https://brew.sh)

1. [Node.js](https://nodejs.org/en/download/package-manager/)

   ```bash
   brew install node
   ```

1. [Rust](https://rustup.rs)

### Windows

1. [Chocolatey](https://chocolatey.org/install)

1. [Node.js](https://nodejs.org/en/download/package-manager/)

   Open Windows PowerShell as an administrator, and run

   ```ps
   cinst nodejs
   ```

1. Build tools

   Open Windows PowerShell as an administrator, and run

   ```ps
   cinst git msys2
   ```

   Open Windows PowerShell as a normal user, and run

   ```ps
   pacman -S diffutils m4 make mingw-w64-x86_64-clang mingw-w64-x86_64-gcc
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

   See [Windows - The rustup book](https://rust-lang.github.io/rustup/installation/windows.html) for details.

### Ubuntu

1. Command line tools and libraries

   ```bash
   sudo apt update
   sudo apt upgrade -y
   sudo apt install -y build-essential curl git libclang-dev m4
   ```

   [libraries required to run Electron](https://github.com/electron/electron/issues/26673):

   ```bash
   sudo apt install -y libatk-bridge2.0-0 libatk1.0-0 libgbm1 libgconf-2-4 libgdk-pixbuf2.0-0 libgtk-3-0 libnss3
   ```

1. [Node.js](https://nodejs.org/en/download/package-manager/)

   ```bash
   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
   sudo apt install -y nodejs
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

   On **Windows**, you need to add MSYS/MinGW **at the beginning** of PATH before executing build commands:

   ```ps1
   $env:PATH = "C:\tools\msys64\usr\bin;C:\tools\msys64\mingw64\bin;" + $env:PATH
   ```
