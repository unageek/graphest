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
