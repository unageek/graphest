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
- ![Blue](images/blue.png) There may or may not be solutions in the pixel.
- ![White](images/white.png) There are no solutions in the pixel.

"Solution" here means any point that satisfies the relation.

## Syntax

### Expression

| Input                                   | Interpreted as                                               | Notes                                                        |
| --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `123`<br />`123.5`<br />`.5`            | 123<br />123.5<br />0.5                                      | A decimal constant.                                          |
| `e`                                     | e                                                            | The base of natural logarithms.                              |
| `pi`                                    | π                                                            |                                                              |
| `-x`                                    | −*x*                                                         |                                                              |
| `x + y`                                 | _x_ + _y_                                                    |                                                              |
| `x - y`                                 | _x_ − _y_                                                    |                                                              |
| `x * y`                                 | _x_ _y_                                                      |                                                              |
| `x / y`                                 | _x_ / _y_                                                    | Undefined for _y_ = 0.                                       |
| `sqrt(x)`                               | √*x*                                                         | Undefined for _x_ < 0.                                       |
| `x^y`                                   | _x_<sup>_y_</sup>                                            | `^` is right-associative: `x^y^z` is equivalent to `x^(y^z)`.<br />See [About Exponentiation](#about-exponentiation) for the definition.<br />See also `exp`, `exp2` and `exp10`. |
| `exp(x)`<br />`exp2(x)`<br />`exp10(x)` | e<sup>_x_</sup><br />2<sup>_x_</sup><br />10<sup>_x_</sup>   |                                                              |
| `log(x)`<br />`log2(x)`<br />`log10(x)` | log<sub>e</sub> _x_<br />log<sub>2</sub> _x_<br />log<sub>10</sub> _x_ | Undefined for _x_ ≤ 0.                                       |
| `sin(x)`                                | sin _x_                                                      |                                                              |
| `cos(x)`                                | cos _x_                                                      |                                                              |
| `tan(x)`                                | tan _x_                                                      | Undefined for _x_ = (_n_ + 1/2)π for all integer _n_.        |
| `asin(x)`                               | sin<sup>−1</sup> _x_                                         | Undefined for _x_ < −1 and _x_ > 1.<br />The range is [−π/2, π/2]. |
| `acox(x)`                               | cos<sup>−1</sup> _x_                                         | Undefined for _x_ < −1 and _x_ > 1.<br />The range is [0, π]. |
| `atan(x)`                               | tan<sup>−1</sup> _x_                                         | The range is (−π/2, π/2).                                    |
| `atan2(y, x)`                           | tan<sup>−1</sup>(_y_ / _x_)                                  | [The two-argument arctangent.](https://en.wikipedia.org/wiki/Atan2)<br />Undefined for (_x_, _y_) = (0, 0).<br />The range is (−π, π]. |
| `sinh(x)`                               | sinh _x_                                                     |                                                              |
| `cosh(x)`                               | cosh _x_                                                     |                                                              |
| `tanh(x)`                               | tanh _x_                                                     |                                                              |
| `asinh(x)`                              | sinh<sup>−1</sup> _x_                                        |                                                              |
| `acosh(x)`                              | cosh<sup>−1</sup> _x_                                        | Undefined for _x_ < 1.<br />The range is [0, ∞).             |
| `atanh(x)`                              | tanh<sup>−1</sup> _x_                                        | Undefined for _x_ ≤ −1 and _x_ ≥ 1.                          |
| `abs(x)`                                | \|_x_\|                                                      |                                                              |
| `min(x, y)`                             | min {_x_, _y_}                                               |                                                              |
| `max(x, y)`                             | max {_x_, _y_}                                               |                                                              |
| `floor(x)`                              | ⌊_x_⌋                                                        | [The floor function.](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions) |
| `ceil(x)`                               | ⌈_x_⌉                                                        | [The ceiling function.](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions) |
| `sign(x)`                               | sgn(_x_)                                                     | [The sign function.](https://en.wikipedia.org/wiki/Sign_function) |
| `mod(x, y)`                             | _x_ mod _y_                                                  | [The modulo operation.](https://en.wikipedia.org/wiki/Modulo_operation)<br />The result is nonnegative, _i.e._, 0 ≤ _x_ mod _y_ < \|_y_\|. |

### Relation

| Input      | Interpreted as | Notes                                                                                                          |
| ---------- | -------------- | -------------------------------------------------------------------------------------------------------------- |
| `x == y`   | _x_ = _y_      |                                                                                                                |
| `x < y`    | _x_ < _y_      |                                                                                                                |
| `x <= y`   | _x_ ≤ _y_      |                                                                                                                |
| `x > y`    | _x_ > _y_      |                                                                                                                |
| `x >= y`   | _x_ ≥ _y_      |                                                                                                                |
| `X && Y`   | _X_ ∧ _Y_      | [Logical conjunction.](https://en.wikipedia.org/wiki/Logical_conjunction)<br />`X` and `Y` must be a relation. |
| `X \|\| Y` | _X_ ∨ _Y_      | [Logical disjunction.](https://en.wikipedia.org/wiki/Logical_disjunction)<br />`X` and `Y` must be a relation. |

You can group a part of an expression or a relation with `(` … `)`.

## Details

Currently, the following algorithms from [Tup01] are implemented: 1.1–3.2, 3.4.1 and 3.4.2.

#### About Exponentiation

To be consistent with GrafEq, the following definitions of exponentiation is implemented.

- For _x_ < 0, _x_<sup>_y_</sup> is defined if and only if _y_ is a rational number with an odd denominator:
  - For any positive integers _m_ and _n_, _x_<sup>±_m_ / _n_</sup> := (<sup>_n_</sup>√_x_)<sup>±_m_</sup>, where <sup>_n_</sup>√_x_ is the real-valued *n*th root of _x_.
  - _x_<sup>±_m_ / _n_</sup> is an even (odd) function of _x_ if _m_ is even (odd).

- 0<sup>0</sup> := 1.

### References

- [Ped] Pedagoguery Software Inc. GrafEq™. http://www.peda.com/grafeq
- [Tup96] Jeffrey Allen Tupper. _Graphing Equations with Generalized Interval Arithmetic._ Master's thesis, University of Toronto, 1996. http://www.dgp.toronto.edu/~mooncake/thesis.pdf
- [Tup01] Jeff Tupper. _Reliable Two-Dimensional Graphing Methods for Mathematical Formulae with Two Free Variables._ SIGGRAPH 2001 Conference Proceedings, 2001. http://www.dgp.toronto.edu/~mooncake/papers/SIGGRAPH2001_Tupper.pdf
