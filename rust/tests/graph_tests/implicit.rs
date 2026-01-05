t!(
    t_7a39ab5463814093913a2ab0ae8ba0bc,
    "1 < gcd(x, y) < 2",
    @timeout(4000),
);

t!(
    t_84a2738be35745da97a9b120c563b4b3,
    "y = t",
    @timeout(8000),
);

t!(
    t_7f25b9f2bd3746b3ae8f95a82921de2b,
    "mod(cos(n/12 π) x + sin(n/12 π) y, 3) = 0",
    @timeout(2000),
);

t!(
    t_37b042e4a46346fda37c9af4b71fb404,
    "⌊16/(2π) ln(r)⌋ = ⌊16/(2π) θ⌋ + n ∧ mod(n, 2) = 0 ∧ 0 ≤ n < 16",
    @timeout(6000),
);

t!(
    t_7f564d75d3de45faa732ffdcec3f7ea5,
    "θ = 2π n/12 ∧ 0 ≤ n < 12",
    @timeout(2000),
);

// Non-square
t!(
    t_3d0c421d8fed4ea58e286d3bf1b1fb3a,
    "r = 8 ∧ 1 < θ < 5",
    @size(456, 789)
);
t!(
    t_4cf5b14b6b0946ff80ac77ce5c4cb165,
    "r = 8 ∧ 1 < θ < 5",
    @size(789, 456)
);

// Check if MPFR's sin/cos is used instead of the Arb's functions around extrema.
t!(
    t_250a7184c7444a3a928fdcb92bc75bf9,
    "sin(x) = cos(y)",
    @bounds(4.712, 4.713, 3.141, 3.142)
);

// Huge coordinates
t!(
    t_2ccd8bdd9d2840bfbaee9b34138011fd,
    "2 y = 2 sin(x)",
    @bounds(1000000000000000, 1000000000000004, -2, 2)
);
