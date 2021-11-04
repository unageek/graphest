// From Examples.md

t!(
    t_b8b04a37eaf64d0491cc2b1e6ee7bb1b,
    "(2y-x-1)(2y-x+1)(2x+y-1)(2x+y+1)((5x-2)^2+(5y-6)^2-10)((5x)^2+(5y)^2-10)((5x+2)^2+(5y+6)^2-10) = 0",
    @bounds(-3, 3, -3, 3),
);

t!(
    t_0b3b446edb104b2680bdb05fcdbef602,
    "((x-2)^2+(y-2)^2-0.4)((x-2)^2+(y-1)^2-0.4)((x-2)^2+y^2-0.4)((x-2)^2+(y+1)^2-0.4)((x-2)^2+(y+2)^2-0.4) ((x-1)^2+(y-2)^2-0.4)((x-1)^2+(y-1)^2-0.4)((x-1)^2+y^2-0.4)((x-1)^2+(y+1)^2-0.4)((x-1)^2+(y+2)^2-0.4) (x^2+(y-2)^2-0.4)(x^2+(y-1)^2-0.4)(x^2+y^2-0.4)(x^2+(y+1)^2-0.4)(x^2+(y+2)^2-0.4) ((x+1)^2+(y-2)^2-0.4)((x+1)^2+(y-1)^2-0.4)((x+1)^2+y^2-0.4)((x+1)^2+(y+1)^2-0.4)((x+1)^2+(y+2)^2-0.4) ((x+2)^2+(y-2)^2-0.4)((x+2)^2+(y-1)^2-0.4)((x+2)^2+y^2-0.4)((x+2)^2+(y+1)^2-0.4)((x+2)^2+(y+2)^2-0.4) = 0",
    @bounds(-3, 3, -3, 3),
);

// Irrationally Contin
t!(
    t_c130725c25914756aa959dcbad0edc87,
    "y = gcd(x, 1)",
    @timeout(3000),
);

// Parabolic Waves
t!(
    t_23ac7cf5b57d4c7388397379cd277762,
    "|sin(sqrt(x^2 + y^2))| = |cos(x)|",
);

// Prime Bars
t!(
    t_402ef2fd017a41d088525e08fb27e03c,
    "gcd(⌊x⌋, Γ(⌊sqrt(2⌊x⌋) + 1/2⌋)) ≤ 1 < x - 1",
    @bounds(0, 40, -20, 20),
);

// Pythagorean Pairs
t!(
    t_ba0d2f30668b421484f34cf614f52744,
    "⌊x⌋^2 + ⌊y⌋^2 = ⌊sqrt(⌊x⌋^2 + ⌊y⌋^2)⌋^2",
    @bounds(-40, 40, -40, 40),
);

// Pythagorean Triples
t!(t_df0cd3aa1f02458689e59c6d5b383da7, "⌊x⌋^2 + ⌊y⌋^2 = 25");

// Rational Beams
t!(
    t_305f167fd5ba4d0e8b6c5408f01cdab6,
    "gcd(x, y) > 1",
    @bounds(0.3, 8.9, 0.4, 9.0),
    @timeout(5000),
);

// Infinite Frequency
t!(t_f2c337b79a9843aa898fb494c3869916, "y = sin(40 / x)");

// O Spike
t!(
    t_0fcfd060295c411d8724a7872c071315,
    "(x (x-3) / (x-3.001))^2 + (y (y-3) / (y-3.001))^2 = 81",
    @timeout(2000),
);

// Solid Disc
t!(
    t_8c9209237ba04e67a715382c7bcbf5e0,
    "81 - x^2 - y^2 = |81 - x^2 - y^2|",
    @timeout(3000),
);

// Spike
t!(
    t_ad4600f870c24171a0bb88f0dfb002e9,
    "y = x (x-3) / (x-3.001)",
);

// Step
t!(t_f8c6fb04be6f464ebb4f12ff2bb52638, "y = atan(9^9^9 (x-1))");

// Upper Triangle
t!(
    t_d2a0e9e88c41406a80a266d78f6dc7a5,
    "x + y = |x + y|",
    @timeout(2000),
);

// Wave
t!(t_95e4d99eaafc4e749ca0703ff6a8f1b1, "y = sin(x) / x");

// binary naturals
t!(
    t_0f59284bfc0948758389bb0f6f91b68a,
    "(1 + 99 ⌊mod(⌊y⌋ 2^⌈x⌉, 2)⌋) (mod(x,1) - 1/2)^2 + (mod(y,1) - 1/2)^2 = 0.15 && ⌊-log(2,y)⌋ < x < 0",
    @bounds(-15, 5, -5, 15),
);

// binary squares
t!(
    t_a85d65f32c5d4ae2831eeccb240071b8,
    "(1 + 99 ⌊mod(⌊y⌋^2 2^⌈x⌉, 2)⌋) (mod(x,1) - 1/2)^2 + (mod(y,1) - 1/2)^2 = 0.15 && x < 0 < ⌊y⌋^2 ≥ 2^-⌈x⌉",
    @bounds(-15, 5, -5, 15),
);

// decimal squares
t!(
    t_23d33db665b843368c9272075452f04b,
    "(mod(892 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(y,1) - 1/2|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 1/2| - 1/4) < 1 || mod(365 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(y,1) - 1/10|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 1/10| - 1/4) < 1 || mod(941 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(y,1) - 9/10|, |mod(x,0.8)+0.1 - 1/2| + |mod(y,1) - 9/10| - 1/4) < 1 || mod(927 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(x,0.8) + 0.1 - 4/5|, |mod(y,1) - 7/10| + |mod(x,0.8) + 0.1 - 4/5| - 1/8) < 1 || mod(881 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(x,0.8) + 0.1 - 1/5|, |mod(y,1) - 7/10| + |mod(x,0.8) + 0.1 - 1/5| - 1/8) < 1 || mod(325 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(x,0.8) + 0.1 - 1/5|, |mod(y,1) - 3/10| + |mod(x,0.8) + 0.1 - 1/5| - 1/8) < 1 || mod(1019 2^-⌊mod(⌊y⌋^2 / 10^-⌈1.25x⌉, 10)⌋, 2) ≥ 1 && 30 max(|mod(x,0.8) + 0.1 - 4/5|, |mod(y,1) - 3/10| + |mod(x,0.8) + 0.1 - 4/5| - 1/8) < 1) && x < 0 < ⌊y⌋^2 ≥ 10^-⌈1.25x⌉",
    @bounds(-7, 3, 1, 11),
);

// bi-infinite binary tree
t!(
    t_ed8070659e554cc881b864687658ee9b,
    "sin(2^⌊y⌋ x + π/4 (y - ⌊y⌋) - π/2) = 0 || sin(2^⌊y⌋ x - π/4 (y - ⌊y⌋) - π/2) = 0",
    @timeout(5000),
);

// Simply Spherical
t!(
    t_7c22e4322e2c4388adf50668b15cadb0,
    "sin(20x) - cos(20y) + 2 > 4 (3/4 - 1/15 sqrt((x+4)^2 + (y-3)^2)) && (x+1)^2 + (y-1)^2 < 25 || sin(20x) - cos(20y) + 2 > 4 (0.65 + 1/π atan(6 (sqrt((x-1)^2/30 + (y+1)^2/9) - 1))) && (x+1)^2 + (y-1)^2 > 25",
    @timeout(2000),
);

// Tube
t!(
    t_bfd842225b5a499bba4be9a08feecd2a,
    "cos(5x) + cos(5/2 (x - sqrt(3) y)) + cos(5/2 (x + sqrt(3) y)) > 1 + 3/2 sin(1/4 sqrt((x+3)^2 + 2 (y-3)^2)) && (x^2 + 2y^2 - 1600) (x^2 + 3 (y-2)^2 - 700) ≤ 0 || cos(5x) + cos(5/2 (x - sqrt(3) y)) + cos(5/2 (x + sqrt(3) y)) > 1 + 2 atan(1/8 sqrt(4 (x-2)^2 + 10 (y+4)^2) - 9)^2 && (x^2 + 2y^2 - 1600) (x^2 + 3 (y-2)^2 - 700) > 0",
    @bounds(-50, 50, -50, 50),
    @timeout(4000),
);

// Frontispiece #2
t!(
    t_70f11f255fe54073996223eec75840f4,
    "x / cos(x) + y / cos(y) = x y / cos(x y) || x / cos(x) + y / cos(y) = -(x y / cos(x y)) || x / cos(x) - y / cos(y) = x y / cos(x y) || x / cos(x) - y / cos(y) = -(x y / cos(x y))",
    @timeout(5000),
);

// Frontispiece
t!(
    t_29f17030e314469e816e03979778f244,
    "x / sin(x) + y / sin(y) = x y / sin(x y) || x / sin(x) + y / sin(y) = -(x y / sin(x y)) || x / sin(x) - y / sin(y) = x y / sin(x y) || x / sin(x) - y / sin(y) = -(x y / sin(x y))",
    @timeout(6000),
);

// Highwire
t!(
    t_564aaa5e88a54895b1851a1f1e5ffa3c,
    "|x cos(x) - y sin(y)| = |x cos(y) - y sin(x)|",
    @bounds(-10.1, 9.9, -9.8, 10.2),
    @timeout(9000),
);

// Trapezoidal Fortress
t!(
    t_b0175d3b58ed46158ba72bbd85670fc9,
    "|x cos(x) + y sin(y)| = x cos(y) - y sin(x)",
    @bounds(-10.1, 9.9, -9.8, 10.2),
    @timeout(8000),
);

// Sharp Threesome
t!(
    t_2bd34182bf58454bb6b7ce7d72548588,
    "sin(sqrt((x+5)^2 + y^2)) cos(8 atan(y / (x+5))) sin(sqrt((x-5)^2 + (y-5)^2)) cos(8 atan((y-5) / (x-5))) sin(sqrt(x^2 + (y+5)^2)) cos(8 atan((y+5) / x)) > 0",
    @timeout(6000),
);

// The Disco Hall
t!(
    t_8b36e2e711b7491f9af53daf696053a9,
    "sin(|x + y|) > max(cos(x^2), sin(y^2))",
    @timeout(2000),
);

// “the patterned star”
t!(
    t_40c29ccd39684aa7847dc32caf5bcab6,
    "0.15 > |rankedMin([cos(8y), cos(4(y-sqrt(3)x)), cos(4(y+sqrt(3)x))], 2) - cos(⌊3/π mod(atan2(y,x),2π) - 0.5⌋) - 0.1| && rankedMin([|2x|, |x-sqrt(3)y|, |x+sqrt(3)y|], 2) < 10",
    @bounds(-10.1, 9.9, -9.8, 10.2),
    @timeout(4000),
);
