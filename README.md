# Graphest

[![build](https://img.shields.io/github/workflow/status/unageek/graphest/build/master)](https://github.com/unageek/graphest/actions?query=branch%3Amaster+workflow%3Abuild)

<p align="center">
  <img src="docs/cover.png">
</p>

Graphest is a graphing calculator that can faithfully plot arbitrary mathematical relations of _x_ and _y_.

## Getting Started

A prebuilt version of the app for Macs with Apple silicon is available on the [releases](https://github.com/unageek/graphest/releases) page.

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

1. Command line tools

   ```bash
   sudo apt install -y build-essential curl git m4
   ```

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

## Algorithm

Graphest aims to provide a modern, open-source and extensible alternative to [GrafEq™](http://www.peda.com/grafeq/) program [Ped].

The following algorithms in [Tup01] are implemented: 1.1–3.2, 3.3 (partially), 3.4.1–3.4.3.

- [Ped] Pedagoguery Software Inc. GrafEq™. http://www.peda.com/grafeq
- [Tup96] Jeffrey Allen Tupper. _Graphing Equations with Generalized Interval Arithmetic._ Master's thesis, University of Toronto, 1996. http://www.dgp.toronto.edu/~mooncake/thesis.pdf
- [Tup01] Jeff Tupper. _Reliable Two-Dimensional Graphing Methods for Mathematical Formulae with Two Free Variables._ SIGGRAPH 2001 Conference Proceedings, 2001. http://www.dgp.toronto.edu/~mooncake/papers/SIGGRAPH2001_Tupper.pdf
