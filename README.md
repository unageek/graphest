# Graphest

## Build

### Install prerequisites

#### macOS

1. Install Command Line Tools for Xcode

   ```bash
   xcode-select --install
   ```

1. [Install Rust](https://rustup.rs)

1. [Install Homebrew](https://brew.sh)

1. Install Node.js

   ```bash
   brew install node
   ```

#### Ubuntu or WSL

1. [Install Rust](https://rustup.rs)

1. Install Node.js and npm

   ```bash
   sudo apt install -y nodejs npm
   ```

### Build the app

1. Clone the repo and install dependencies

   ```bash
   git clone https://github.com/unageek/inari-graph-app.git
   cd inari-graph-app
   npm install
   ```

1. Build and run

   ```bash
   npm run build
   npm start
   ```
