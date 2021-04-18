# inari-graph

[![build](https://img.shields.io/github/workflow/status/unageek/inari-graph/build/master)](https://github.com/unageek/inari-graph/actions?query=branch%3Amaster+workflow%3Abuild)

<p align="center">
  <img src="images/cover.gif"><br>
  The graph of sin(<i>x</i> ± sin <i>y</i>) (sin <i>x</i> ± <i>y</i>) = cos(sin((sin <i>x</i> ± cos <i>y</i>) (sin <i>y</i> ± cos <i>x</i>))) over [4, 6.5] × [2, 4.5].
</p>

inari-graph can plot the graph of an arbitrary relation (like above) in a reliable manner. It aims to provide an open-source and extensible alternative to [GrafEq™](http://www.peda.com/grafeq/) program [Ped].

## Usage

If you are running Windows, [install Ubuntu on WSL](https://ubuntu.com/wsl) and follow the steps below.

1. Install Rust and other build tools

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   sudo apt install build-essential curl git m4
   ```

1. Build

   ```bash
   git clone https://github.com/unageek/inari-graph.git
   cd inari-graph
   cargo build --release
   ```

   You can optionally supply `--features "arb"` option to boost plotting performance and enable [additional functions](#special-functions). In this case, the build can take a long time (~10 minutes).

   ```bash
   cargo build --release --features "arb"
   ```

1. Run

   ```bash
   ./target/release/inari-graph "y = sin(x)"
   ```

   The plot will be saved to `graph.png` in the current directory.

   Use `-h` option to view help:


## Color Legend

- ![Black](images/black.png) There is at least one solution in the pixel.
- ![Blue](images/blue.png) There may or may not be solutions in the pixel.
- ![White](images/white.png) There are no solutions in the pixel.

"Solution" here means any point that satisfies the relation.

## References

- [Ped] Pedagoguery Software Inc. GrafEq™. http://www.peda.com/grafeq
- [Tup96] Jeffrey Allen Tupper. *Graphing Equations with Generalized Interval Arithmetic.* Master's thesis, University of Toronto, 1996. http://www.dgp.toronto.edu/~mooncake/thesis.pdf
- [Tup01] Jeff Tupper. *Reliable Two-Dimensional Graphing Methods for Mathematical Formulae with Two Free Variables.* SIGGRAPH 2001 Conference Proceedings, 2001. http://www.dgp.toronto.edu/~mooncake/papers/SIGGRAPH2001_Tupper.pdf

