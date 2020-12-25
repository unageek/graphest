# inari-graph

[![build](https://img.shields.io/github/workflow/status/mizuno-gsinet/inari-graph/build/master)](https://github.com/mizuno-gsinet/inari-graph/actions?query=branch%3Amaster+workflow%3Abuild)

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

   You can optionally supply `--features "arb"` option to boost plotting performance and enable [additional functions](#special-functions). In this case, the build can take a long time (~10 minutes).

   ```bash
   cargo build --release --features "arb"
   ```

1. Run

   ```bash
   ./target/release/inari-graph "y = sin(x)"
   ```

   The plot will be saved to `graph.png` in the current directory.

   Try plotting some [example relations](Examples.md) or your own ones.

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

| Input                                  | Interpreted as                                               | Details                                                      |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `123`<br />`123.5`<br />`.5`           | 123<br />123.5<br />0.5                                      |                                                              |
| `e`                                    | e                                                            | The base of natural logarithms.                              |
| `pi`<br />`π`                          | π                                                            |                                                              |
| `-x`                                   | −*x*                                                         |                                                              |
| `x + y`                                | *x* + *y*                                                    |                                                              |
| `x - y`                                | *x* − *y*                                                    |                                                              |
| `x y`<br />`x * y`                     | *x* *y*                                                      |                                                              |
| `x / y`                                | *x* / *y*                                                    | Undefined for *y* = 0.                                       |
| `sqrt(x)`                              | √*x*                                                         | Undefined for *x* < 0.                                       |
| `x ^ y`                                | *x*<sup>*y*</sup>                                            | `^` is right-associative: `x^y^z` is equivalent to `x^(y^z)`.<br />See [About Exponentiation](#about-exponentiation) for the definition. |
| `exp(x)`                               | e<sup>*x*</sup>                                              |                                                              |
| `ln(x)`<br />`log(x)`<br />`log(b, x)` | log<sub>e</sub> *x*<br />log<sub>10</sub> *x*<br />log<sub>*b*</sub> *x* | Undefined for *x* ≤ 0, *b* ≤ 0 and *b* = 1.                  |
| `\|x\|`                                | \|*x*\|                                                      |                                                              |
| `min(x, y)`                            | min {*x*, *y*}                                               |                                                              |
| `max(x, y)`                            | max {*x*, *y*}                                               |                                                              |
| `floor(x)`<br />`⌊x⌋`                  | ⌊*x*⌋                                                        | The [floor function](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions). |
| `ceil(x)`<br />`⌈x⌉`                   | ⌈*x*⌉                                                        | The [ceiling function](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions). |
| `sign(x)`                              | sgn(*x*)                                                     | The [sign function](https://en.wikipedia.org/wiki/Sign_function). |
| `mod(x, y)`                            | *x* mod *y*                                                  | The nonnegative remainder of *x*/*y* ([modulo operation](https://en.wikipedia.org/wiki/Modulo_operation)).<br />0 ≤ *x* mod *y* < \|*y*\|. |
| `gcd(x, y)`                            | gcd(*x*, *y*)                                                | The [greatest common divisor](https://en.wikipedia.org/wiki/Greatest_common_divisor) of *x* and *y*. |
| `lcm(x, y)`                            | lcm(*x*, *y*)                                                | The [least common multiple](https://en.wikipedia.org/wiki/Least_common_multiple) of *x* and *y*. |

See also [Trigonometric Functions](#trigonometric-functions) and [Special Functions](#special-functions).

### Relation

| Input                 | Interpreted as | Details                                                      |
| --------------------- | -------------- | ------------------------------------------------------------ |
| `x = y`               | *x* = *y*      |                                                              |
| `x < y`               | *x* < *y*      |                                                              |
| `x <= y`<br />`x ≤ y` | *x* ≤ *y*      |                                                              |
| `x > y`               | *x* > *y*      |                                                              |
| `x >= y`<br />`x ≥ y` | *x* ≥ *y*      |                                                              |
| `X && Y`              | *X* ∧ *Y*      | [Logical conjunction.](https://en.wikipedia.org/wiki/Logical_conjunction)<br />`X` and `Y` must be a relation. |
| `X \|\| Y`            | *X* ∨ *Y*      | [Logical disjunction.](https://en.wikipedia.org/wiki/Logical_disjunction)<br />`X` and `Y` must be a relation. |

You can group a part of an expression or a relation with `(` … `)`.

## Tips

- You can use [Matplotlib](https://matplotlib.org/) to add a frame to a plot:

  ```py
  #!/usr/bin/env python3
  import matplotlib.pyplot as plt

  image = plt.imread('graph.png')
  fig, ax = plt.subplots(figsize=(5, 5))
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.tick_params(top=True, right=True)
  ax.imshow(image, extent=[-10, 10, -10, 10], interpolation='none')
  fig.tight_layout()
  fig.savefig('graph.pdf')
  ```

- Writing polynomials in [Horner form](https://en.wikipedia.org/wiki/Horner%27s_method) can significantly improve the speed of graphing. For example, the truncated Maclaurin series of `sin(x)` up to order 17 is:

  ```text
  "y = x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 - x^11/39916800 + x^13/6227020800 - x^15/1307674368000 + x^17/355687428096000"
  ```

  The horner form of the above is:

  ```text
  "y = x (1 + x^2 (-1/6 + x^2 (1/120 + x^2 (-1/5040 + x^2 (1/362880 + x^2 (-1/39916800 + x^2 (1/6227020800+x^2 (-1/1307674368000 + x^2/355687428096000))))))))"
  ```

## Details

Currently, the following algorithms from [Tup01] are implemented: 1.1–3.2, 3.4.1–3.4.3.

### About Exponentiation

To be consistent with GrafEq, the following definitions of exponentiation is implemented.

- For *x* < 0, *x*<sup>*y*</sup> is defined if and only if *y* is a rational number with an odd denominator:
  - For any positive integers *m* and *n*, *x*<sup>±*m* / *n*</sup> := (<sup>*n*</sup>√*x*)<sup>±*m*</sup>, where <sup>*n*</sup>√*x* is the real-valued *n*th root of *x*.
  - *x*<sup>±*m* / *n*</sup> is an even (odd) function of *x* if *m* is even (odd).
- 0<sup>0</sup> := 1.

## Function Reference

### Trigonometric Functions

| Input         | Interpreted as              | Details                                                      |
| ------------- | --------------------------- | ------------------------------------------------------------ |
| `sin(x)`      | sin *x*                     |                                                              |
| `cos(x)`      | cos *x*                     |                                                              |
| `tan(x)`      | tan *x*                     | Undefined for *x* = (*n* + 1/2)π for all integers *n*.       |
| `asin(x)`     | sin<sup>−1</sup> *x*        | Undefined for *x* < −1 and *x* > 1.<br />The range is [−π/2, π/2]. |
| `acox(x)`     | cos<sup>−1</sup> *x*        | Undefined for *x* < −1 and *x* > 1.<br />The range is [0, π]. |
| `atan(x)`     | tan<sup>−1</sup> *x*        | The range is (−π/2, π/2).                                    |
| `atan2(y, x)` | tan<sup>−1</sup>(*y* / *x*) | The [two-argument arctangent](https://en.wikipedia.org/wiki/Atan2).<br />Undefined for (*x*, *y*) = (0, 0).<br />The range is (−π, π]. |
| `sinh(x)`     | sinh *x*                    |                                                              |
| `cosh(x)`     | cosh *x*                    |                                                              |
| `tanh(x)`     | tanh *x*                    |                                                              |
| `asinh(x)`    | sinh<sup>−1</sup> *x*       |                                                              |
| `acosh(x)`    | cosh<sup>−1</sup> *x*       | Undefined for *x* < 1.<br />The range is [0, ∞).             |
| `atanh(x)`    | tanh<sup>−1</sup> *x*       | Undefined for *x* ≤ −1 and *x* ≥ 1.                          |

### Special Functions

| Input                  | Interpreted as | Details                                                      |
| ---------------------- | -------------- | ------------------------------------------------------------ |
| `Gamma(x)`<br />`Γ(x)` | Γ(*x*)         | The [gamma function](https://en.wikipedia.org/wiki/Gamma_function).<br />Undefined for *x* = 0, −1, −2, … |
| `erf(x)`               | erf(*x*)       | The [error function](https://en.wikipedia.org/wiki/Error_function). |
| `erfc(x)`              | erfc(*x*)      | The complementary error function.                            |

Functions that require building with `--features "arb"` option:

| Input                                            | Interpreted as                                   | Details                                                      |
| ------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------ |
| `Gamma(a, x)`<br />`Γ(a, x)`                     | Γ(*a*, *x*)                                      | The [upper incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).<br />*a* must be an exact number.<sup>1</sup> |
| `erfi(x)`                                        | erfi(*x*)                                        | The imaginary error function.                                |
| `Ei(x)`                                          | Ei(*x*)                                          | The [exponential integral](https://en.wikipedia.org/wiki/Exponential_integral). |
| `li(x)`                                          | li(*x*)                                          | The [logarithmic integral](https://en.wikipedia.org/wiki/Logarithmic_integral_function). |
| `Si(x)`                                          | Si(*x*)                                          | The [sine integral](https://en.wikipedia.org/wiki/Trigonometric_integral). |
| `Ci(x)`                                          | Ci(*x*)                                          | The cosine integral.                                         |
| `Shi(x)`                                         | Shi(*x*)                                         | The hyperbolic sine integral.                                |
| `Chi(x)`                                         | Chi(*x*)                                         | The hyperbolic cosine integral.                              |
| `S(x)`<br />`C(x)`                               | S(*x*)<br />C(*x*)                               | The [Fresnel integrals](https://en.wikipedia.org/wiki/Fresnel_integral). |
| `J(n, x)`<br />`Y(n, x)`                         | J<sub>*n*</sub>(*x*)<br />Y<sub>*n*</sub>(*x*)   | The [Bessel functions](https://en.wikipedia.org/wiki/Bessel_function).<br />*n* must be an integer or a half-integer. |
| `I(n, x)`<br />`K(n, x)`                         | I<sub>*n*</sub>(*x*)<br />K<sub>*n*</sub>(*x*)   | The modified Bessel functions.<br />*n* must be an integer or a half-integer. |
| `Ai(x)`<br />`Bi(x)`<br />`Ai'(x)`<br />`Bi'(x)` | Ai(*x*)<br />Bi(*x*)<br />Ai′(*x*)<br />Bi′(*x*) | The [Airy functions](https://en.wikipedia.org/wiki/Airy_function) and their derivatives. |

<sup>1.</sup> A number that can be represented as a double-precision floating-point number, such as 1.5 or −3.0625.

## References

- [Ped] Pedagoguery Software Inc. GrafEq™. http://www.peda.com/grafeq
- [Tup96] Jeffrey Allen Tupper. *Graphing Equations with Generalized Interval Arithmetic.* Master's thesis, University of Toronto, 1996. http://www.dgp.toronto.edu/~mooncake/thesis.pdf
- [Tup01] Jeff Tupper. *Reliable Two-Dimensional Graphing Methods for Mathematical Formulae with Two Free Variables.* SIGGRAPH 2001 Conference Proceedings, 2001. http://www.dgp.toronto.edu/~mooncake/papers/SIGGRAPH2001_Tupper.pdf

