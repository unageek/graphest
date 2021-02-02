# Examples

🐌: Takes a long time to terminate.

## Basic Examples

```text
"sin(x) = cos(y)"
```

```text
"y - x = sin(exp(x + y))"
```

```text
"x^2 + y^2 = 1 || y = -cos(x)"
```

## Algebraic Equations

```text
"(2y-x-1)(2y-x+1)(2x+y-1)(2x+y+1)((5x-2)^2+(5y-6)^2-10)((5x)^2+(5y)^2-10)((5x+2)^2+(5y+6)^2-10)=0" \
-b -3 3 -3 3
```

```text
"((x-2)^2+(y-2)^2-0.4)((x-2)^2+(y-1)^2-0.4)((x-2)^2+y^2-0.4)((x-2)^2+(y+1)^2-0.4)((x-2)^2+(y+2)^2-0.4) \
 ((x-1)^2+(y-2)^2-0.4)((x-1)^2+(y-1)^2-0.4)((x-1)^2+y^2-0.4)((x-1)^2+(y+1)^2-0.4)((x-1)^2+(y+2)^2-0.4) \
 (x^2+(y-2)^2-0.4)(x^2+(y-1)^2-0.4)(x^2+y^2-0.4)(x^2+(y+1)^2-0.4)(x^2+(y+2)^2-0.4) \
 ((x+1)^2+(y-2)^2-0.4)((x+1)^2+(y-1)^2-0.4)((x+1)^2+y^2-0.4)((x+1)^2+(y+1)^2-0.4)((x+1)^2+(y+2)^2-0.4) \
 ((x+2)^2+(y-2)^2-0.4)((x+2)^2+(y-1)^2-0.4)((x+2)^2+y^2-0.4)((x+2)^2+(y+1)^2-0.4)((x+2)^2+(y+2)^2-0.4) = 0" \
-b -3 3 -3 3
```

## Examples taken from [GrafEq](http://www.peda.com/grafeq/)

- 📂 Single Relation/Abstract/Simple/

  - 📄 Irrationally Contin.gqs 🐌🐌🐌

    ```test
    "y = gcd(x, 1)" -b -0.68 3.14 -0.68 3.14
    ```

  - 📄 Parabolic Waves.gqs

    ```text
    "|sin(sqrt(x^2 + y^2))| = |cos(x)|"
    ```

  - 📄 Prime Bars.gqs

    ```text
    "gcd(⌊x⌋, Γ(⌊sqrt(2⌊x⌋) + 1/2⌋)) ≤ 1 < x - 1" -b 0 40 -20 20
    ```

  - 📄 Pythagorean Pairs.gqs

    ```text
    "⌊x⌋^2 + ⌊y⌋^2 = ⌊sqrt(⌊x⌋^2 + ⌊y⌋^2)⌋^2" -b -40 40 -40 40
    ```

  - 📄 Pythagorean Triples.gqs

    ```text
    "⌊x⌋^2 + ⌊y⌋^2 = 25"
    ```

  - 📄 Rational Beams.gqs

    ```text
    "gcd(x, y) > 1" -b 0.3 8.9 0.4 9.0
    ```

- 📂 Single Relation/Abstract/Traditionally Difficult/

  - 📄 Infinite Frequency.gqs

    ```text
    "y = sin(40 / x)"
    ```

  - 📄 O Spike.gqs

    ```text
    "(x (x-3) / (x-3.001))^2 + (y (y-3) / (y-3.001))^2 = 81"
    ```

  - 📄 Solid Disc.gqs

    ```text
    "81 - x^2 - y^2 = |81 - x^2 - y^2|"
    ```

  - 📄 Spike.gqs

    ```text
    "y = x (x-3) / (x-3.001)"
    ```

  - 📄 Step.gqs

    ```text
    "y = atan(9^9^9 (x-1))"
    ```

  - 📄 Upper Triangle.gqs

    ```text
    "x + y = |x + y|"
    ```

  - 📄 Wave.gqs

    ```text
    "y = sin(x) / x"
    ```

- 📂 Single Relation/Enumerations/Binary/

  - 📄 binary naturals.gqs

    ```text
    "(1 + 99 ⌊mod(⌊y⌋ 2^⌈x⌉, 2)⌋) (mod(x,1) - 1/2)^2 + (mod(y,1) - 1/2)^2 = 0.15 && \
     ⌊-log(2,y)⌋ < x < 0" -b -15 5 -5 15
    ```

  - 📄 binary squares.gqs

    ```text
    "(1 + 99 ⌊mod(⌊y⌋^2 2^⌈x⌉, 2)⌋) (mod(x,1) - 1/2)^2 + (mod(y,1) - 1/2)^2 = 0.15 && \
     x < 0 < ⌊y⌋^2 ≥ 2^-⌈x⌉" -b -15 5 -5 15
    ```

- 📂 Single Relation/Enumerations/Decimal/

  - 📄 decimal squares.gqs

    ```text
    "(mod(892 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(y,1) - 1/2|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 1/2| - 1/4) < 1 || \
      mod(365 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(y,1) - 1/10|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 1/10| - 1/4) < 1 || \
      mod(941 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(y,1) - 9/10|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 9/10| - 1/4) < 1 || \
      mod(927 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(x,0.8) + 0.1 - 4/5|, |mod(y,1) - 7/10| + |mod(x,0.8) + 0.1 - 4/5| - 1/8) < 1 || \
      mod(881 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(x,0.8) + 0.1 - 1/5|, |mod(y,1) - 7/10| + |mod(x,0.8) + 0.1 - 1/5| - 1/8) < 1 || \
      mod(325 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(x,0.8) + 0.1 - 1/5|, |mod(y,1) - 3/10| + |mod(x,0.8) + 0.1 - 1/5| - 1/8) < 1 || \
      mod(1019 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && \
      30 max(|mod(x,0.8) + 0.1 - 4/5|, |mod(y,1) - 3/10| + |mod(x,0.8) + 0.1 - 4/5| - 1/8) < 1) && \
     x < 0 < ⌊y⌋^2 ≥ 10^-⌈1.25x⌉" -b -7 3 1 11
    ```

- 📂 Single Relation/Enumerations/Trees/

  - 📄 bi-infinite binary tree.gqs

    ```text
    "sin(2^⌊y⌋ x + π/4 (y - ⌊y⌋) - π/2) = 0 || sin(2^⌊y⌋ x - π/4 (y - ⌊y⌋) - π/2) = 0"
    ```

- 📂 Single Relation/Enumerations/Half-Toned/

  - 📄 Simply Spherical.gqs

    ```text
    "sin(20x) - cos(20y) + 2 > 4 (3/4 - 1/15 sqrt((x+4)^2 + (y-3)^2)) && (x+1)^2 + (y-1)^2 < 25 || \
     sin(20x) - cos(20y) + 2 > 4 (0.65 + 1/π atan(6 (sqrt((x-1)^2/30 + (y+1)^2/9) - 1))) && (x+1)^2 + (y-1)^2 > 25"
    ```

  - 📄 Tube.gqs

    ```text
    "cos(5x) + cos(5/2 (x - sqrt(3) y)) + cos(5/2 (x + sqrt(3) y)) > 1 + 3/2 sin(1/4 sqrt((x+3)^2 + 2 (y-3)^2)) && \
     (x^2 + 2y^2 - 1600) (x^2 + 3 (y-2)^2 - 700) ≤ 0 || \
     cos(5x) + cos(5/2 (x - sqrt(3) y)) + cos(5/2 (x + sqrt(3) y)) > 1 + 2 atan(1/8 sqrt(4 (x-2)^2 + 10 (y+4)^2) - 9)^2 && \
     (x^2 + 2y^2 - 1600) (x^2 + 3 (y-2)^2 - 700) > 0" -b -50 50 -50 50
    ```

- 📂 Single Relation/Linelike/

  - 📄 Frontispiece #2.gqs

    ```text
    "x / cos(x) + y / cos(y) = x y / cos(x y) || x / cos(x) + y / cos(y) = -(x y / cos(x y)) || \
     x / cos(x) - y / cos(y) = x y / cos(x y) || x / cos(x) - y / cos(y) = -(x y / cos(x y))"
    ```

  - 📄 Frontispiece.gqs

    ```text
    "x / sin(x) + y / sin(y) = x y / sin(x y) || x / sin(x) + y / sin(y) = -(x y / sin(x y)) || \
     x / sin(x) - y / sin(y) = x y / sin(x y) || x / sin(x) - y / sin(y) = -(x y / sin(x y))"
    ```

  - 📄 Hair.gqs 🐌

    ```text
    "sin((x + sin(y)) (sin(x) + y)) = cos(sin((sin(x) + cos(y)) (sin(y) + cos(x)))) || \
     sin((x + sin(y)) (sin(x) + y)) = cos(sin((sin(x) + cos(y)) (sin(y) - cos(x)))) || \
     sin((x + sin(y)) (sin(x) + y)) = cos(sin((sin(x) - cos(y)) (sin(y) + cos(x)))) || \
     sin((x + sin(y)) (sin(x) + y)) = cos(sin((sin(x) - cos(y)) (sin(y) - cos(x)))) || \
     sin((x + sin(y)) (sin(x) - y)) = cos(sin((sin(x) + cos(y)) (sin(y) + cos(x)))) || \
     sin((x + sin(y)) (sin(x) - y)) = cos(sin((sin(x) + cos(y)) (sin(y) - cos(x)))) || \
     sin((x + sin(y)) (sin(x) - y)) = cos(sin((sin(x) - cos(y)) (sin(y) + cos(x)))) || \
     sin((x + sin(y)) (sin(x) - y)) = cos(sin((sin(x) - cos(y)) (sin(y) - cos(x)))) || \
     sin((x - sin(y)) (sin(x) + y)) = cos(sin((sin(x) + cos(y)) (sin(y) + cos(x)))) || \
     sin((x - sin(y)) (sin(x) + y)) = cos(sin((sin(x) + cos(y)) (sin(y) - cos(x)))) || \
     sin((x - sin(y)) (sin(x) + y)) = cos(sin((sin(x) - cos(y)) (sin(y) + cos(x)))) || \
     sin((x - sin(y)) (sin(x) + y)) = cos(sin((sin(x) - cos(y)) (sin(y) - cos(x)))) || \
     sin((x - sin(y)) (sin(x) - y)) = cos(sin((sin(x) + cos(y)) (sin(y) + cos(x)))) || \
     sin((x - sin(y)) (sin(x) - y)) = cos(sin((sin(x) + cos(y)) (sin(y) - cos(x)))) || \
     sin((x - sin(y)) (sin(x) - y)) = cos(sin((sin(x) - cos(y)) (sin(y) + cos(x)))) || \
     sin((x - sin(y)) (sin(x) - y)) = cos(sin((sin(x) - cos(y)) (sin(y) - cos(x))))" \
    -b 4.0 6.5 2.0 4.5
    ```

  - 📄 Highwire.gqs

    ```text
    "|x cos(x) - y sin(y)| = |x cos(y) - y sin(x)|"
    ```

  - 📄 Trapezoidal Fortress.gqs

    ```text
    "|x cos(x) + y sin(y)| = x cos(y) - y sin(x)"
    ```

- 📂 Single Relation/Solid/

  - 📄 Sharp Threesome.gqs

    ```text
    "sin(sqrt((x+5)^2 + y^2)) cos(8 atan(y / (x+5))) sin(sqrt((x-5)^2 + (y-5)^2)) \
     cos(8 atan((y-5) / (x-5))) sin(sqrt(x^2 + (y+5)^2)) cos(8 atan((y+5) / x)) > 0"
    ```

  - 📄 The Disco Hall.gqs

    ```text
    "sin(|x + y|) > max(cos(x^2), sin(y^2))"
    ```

## Examples taken from [Tup01]

- Fig. 15 (b) “the patterned star”

  ```text
  "0.15 > |min(cos(8y), cos(4(y-sqrt(3)x)), cos(4(y+sqrt(3)x)), rank=2) - \
   cos(⌊3/π mod(atan2(y,x),2π) - 0.5⌋) - 0.1| && \
   min(|2x|, |x-sqrt(3)y|, |x+sqrt(3)y|, rank=2) < 10"
  ```

## Examples taken from [GrafEq Reviews](http://www.peda.com/grafeq/reviews.html)

```text
"y = sqrt(x)^2"
```

```text
"y = sqrt(x-1) / sqrt(x-3)"
```

The graph must be empty:

```text
"y = sqrt(x-3) sqrt(1-x)"
```

## Examples taken from [Cool Graphs of Implicit Equations](https://web.archive.org/web/20160221140058/http://www.xamuel.com/graphs-of-implicit-equations/)

🐌

```text
"exp(sin(x) + cos(y)) = sin(exp(x + y))"
```

```text
"sin(sin(x) + cos(y)) = cos(sin(x y) + cos(x))"
```

```text
"sin(x^2 + y^2) = cos(x y)"
```

```text
"|sin(x^2 - y^2)| = sin(x + y) + cos(x y)"
```

```text
"|sin(x^2 + 2 x y)| = sin(x - 2 y)"
```

## Tests for Conjunction and Disjunction

The graph must be empty:

```text
"y = x && y = x + 0.0001"
```

The graph must not be empty:

```text
"y = x || y = x + 0.0001"
```

```text
'y < sqrt(x) && y < sqrt(-x)'
```

```text
'y < sqrt(x) || y < sqrt(-x)'
```

```text
"y = sin(40 / x) && (x > 0 && y > 0)"
```

```text
"y = sin(40 / x) && (x > 0 || y > 0)"
```

```text
"y = sin(40 / x) || (x > 0 && y > 0)"
```

```text
"y = sin(40 / x) || (x > 0 || y > 0)"
```
