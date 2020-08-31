# inari-graph

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
   git clone https://github.com/mizuno-gsinet/inari-graph.git
   cd inari-graph
   cargo build --release
   ```

1. Run

   ```bash
   ./target/release/inari-graph "y == sin(x)"
   ```

   The plot will be saved to `graph.png` in the current directory.

   Try plotting some [example relations](Examples.md) or your own one.

   Use `-h` option to show help:

   ```bash
   inari-graph 
   Plots the graph of a relation over the x-y plane.
   
   USAGE:
       inari-graph [OPTIONS] [relation]
   
   ARGS:
       <relation>    Relation to plot.
   
   FLAGS:
       -h, --help       Prints help information
       -V, --version    Prints version information
   
   OPTIONS:
       -b <xmin> <xmax> <ymin> <ymax>        Bounds of the plot region. [default: -10 10 -10 10]
       -o <output>                           Output file, only .png is supported. [default: graph.png]
       -s <width> <height>                   Dimensions of the plot. [default: 1024 1024]
   ```

## Color Legend

- ![Black](images/black.png) There is at least one solution in the pixel.
- ![Blue](images/blue.png) The program has not decided the existence or absence of solutions.
- ![White](images/white.png) There are no solutions in the pixel.

"Solution" here means any point that satisfies the relation.

## Syntax

### Expression

| Input                                   | Interpreted as                                               | Notes                                                        |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `-x`                                    | −*x*                                                         |                                                              |
| `x + y`                                 | *x* + *y*                                                    |                                                              |
| `x - y`                                 | *x* − *y*                                                    |                                                              |
| `x * y`                                 | *x* *y*                                                      |                                                              |
| `x / y`                                 | *x* / *y*                                                    | Undefined for *y* = 0.                                       |
| `sqrt(x)`                               | √*x*                                                         | Undefined for *x* < 0.                                       |
| `x^n`                                   | *x*<sup>*n*</sup>                                            | `n` must be an integer constant.<br />Repetition like `x^2^3` is not supported.<br />See also `exp`, `exp2` and `exp10`. |
| `exp(x)`<br />`exp2(x)`<br />`exp10(x)` | e<sup>*x*</sup><br />2<sup>*x*</sup><br />10<sup>*x*</sup>   |                                                              |
| `log(x)`<br />`log2(x)`<br />`log10(x)` | log<sub>e</sub> *x*<br />log<sub>2</sub> *x*<br />log<sub>10</sub> *x* | Undefined for *x* ≤ 0.                                       |
| `sin(x)`                                | sin *x*                                                      |                                                              |
| `cos(x)`                                | cos *x*                                                      |                                                              |
| `tan(x)`                                | tan *x*                                                      | Undefined for *x* = (*n* + 1/2)π for all integer *n*.        |
| `asin(x)`                               | sin<sup>−1</sup> *x*                                         | Undefined for *x* < −1 and *x* > 1.<br />The range is [−π/2, π/2]. |
| `acox(x)`                               | cos<sup>−1</sup> *x*                                         | Undefined for *x* < −1 and *x* > 1.<br />The range is [0, π]. |
| `atan(x)`                               | tan<sup>−1</sup> *x*                                         | The range is (−π/2, π/2).                                    |
| `atan2(y, x)`                           | tan<sup>−1</sup>(*y* / *x*)                                  | [The two-argument arctangent.](https://en.wikipedia.org/wiki/Atan2)<br />Undefined for (*x*, *y*) = (0, 0).<br />The range is (−π, π]. |
| `sinh(x)`                               | sinh *x*                                                     |                                                              |
| `cosh(x)`                               | cosh *x*                                                     |                                                              |
| `tanh(x)`                               | tanh *x*                                                     |                                                              |
| `asinh(x)`                              | sinh<sup>−1</sup> *x*                                        |                                                              |
| `acosh(x)`                              | cosh<sup>−1</sup> *x*                                        | Undefined for *x* < 1.<br />The range is [0, ∞).             |
| `atanh(x)`                              | tanh<sup>−1</sup> *x*                                        | Undefined for *x* ≤ −1 and *x* ≥ 1.                          |
| `abs(x)`                                | \|*x*\|                                                      |                                                              |
| `min(x, y)`                             | min {*x*, *y*}                                               |                                                              |
| `max(x, y)`                             | max {*x*, *y*}                                               |                                                              |
| `floor(x)`                              | ⌊*x*⌋                                                        | [The floor function.](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions) |
| `ceil(x)`                               | ⌈*x*⌉                                                        | [The ceiling function.](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions) |
| `sign(x)`                               | sgn(*x*)                                                     | [The sign function.](https://en.wikipedia.org/wiki/Sign_function) |
| `mod(x, y)`                             | *x* mod *y*                                                  | [The modulo operation.](https://en.wikipedia.org/wiki/Modulo_operation)<br />The result is nonnegative, *i.e.*, 0 ≤ *x* mod *y* < \|*y*\|. |

### Relation

| Input    | Interpreted as | Notes                                                        |
| -------- | -------------- | ------------------------------------------------------------ |
| `x == y` | *x* = *y*      |                                                              |
| `x < y`  | *x* < *y*      |                                                              |
| `x <= y` | *x* ≤ *y*      |                                                              |
| `x > y`  | *x* > *y*      |                                                              |
| `x >= y` | *x* ≥ *y*      |                                                              |
| `X && Y` | *X* ∧ *Y*      | [Logical conjunction.](https://en.wikipedia.org/wiki/Logical_conjunction)<br />`X` and `Y` must be a relation. |
| `X \|\| Y` | *X* ∨ *Y*      | [Logical disjunction.](https://en.wikipedia.org/wiki/Logical_disjunction)<br />`X` and `Y` must be a relation. |

You can group a part of an expression or a relation with `(` … `)`.

## Details

Currently, the following algorithms from [Tup01] are implemented: 1.1–3.2, 3.4.1 and 3.4.2.

## References

- [Ped] Pedagoguery Software Inc. GrafEq™. http://www.peda.com/grafeq
- [Tup96] Jeffrey Allen Tupper. *Graphing Equations with Generalized Interval Arithmetic.* Master's thesis, University of Toronto, 1996. http://www.dgp.toronto.edu/~mooncake/thesis.pdf
- [Tup01] Jeff Tupper. *Reliable Two-Dimensional Graphing Methods for Mathematical Formulae with Two Free Variables.* SIGGRAPH 2001 Conference Proceedings, 2001. http://www.dgp.toronto.edu/~mooncake/papers/SIGGRAPH2001_Tupper.pdf
