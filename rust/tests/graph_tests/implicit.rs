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

// Check if `arb_sin_precise`/`arb_cos_precise` are used properly.
t!(
    t_250a7184c7444a3a928fdcb92bc75bf9,
    "sin(x) = cos(y)",
    @bounds(4.712, 4.713, 3.141, 3.142)
);
