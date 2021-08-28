# Graphest Graphing Engine

<p align="center">
  <img src="images/cover.gif"><br>
  The graph of sin(<i>x</i> ± sin <i>y</i>) (sin <i>x</i> ± <i>y</i>) = cos(sin((sin <i>x</i> ± cos <i>y</i>) (sin <i>y</i> ± cos <i>x</i>))) over [4, 6.5] × [2, 4.5].
</p>

## `graph`

`graph` is the only binary of this crate.

```bash
cargo r --release -- "y = sin(x)"
```

By default:

- The plot will be saved to the file `graph.png` in the current directory.
- The following colors are used:
  - ![Black](images/black.png) The pixel contains a solution; a solution is a point that satisfies the relation.
  - ![Blue](images/blue.png) The pixel may or may not contain a solution.
  - ![White](images/white.png) The pixel does not contain a solution

Use the option `-h` to show usage.

## Conditional Features

- `arb` (enabled by default) - Use [Arb](https://arblib.org), in addition to MPFR, as the underlying implementation of interval functions to speed up evaluation and enable additional functions. With this feature enabled, the unit tests of FLINT and Arb are run during building the crate, which can take quote a long time (~45 minutes). So you might want to skip them by commenting out the statements that contains `.arg("check")` in [build.rs](build.rs).

## Documentation

To build the documentation and open it in the browser, run:

```bash
RUSTDOCFLAGS="-Arustdoc::private_intra_doc_links" cargo doc --lib --open --document-private-items
```
