// Identity.
t!(
    t_54cc8216d22149aaa67c853c772b6477,
    "max(|x|, |y|) = 7",
    @dilate("0,0,0;0,1,0;0,0,0"),
    @size(8, 8)
);

// Duplicate horizontally.
t!(
    t_79611f10496d4bfa8ebe084ffd1240c8,
    "max(|x|, |y|) = 7",
    @dilate("0,0,0;1,0,1;0,0,0"),
    @size(8, 8)
);

// Duplicate vertically.
t!(
    t_b5b6122d0843477581498ca873716f00,
    "max(|x|, |y|) = 7",
    @dilate("0,1,0;0,0,0;0,1,0"),
    @size(8, 8)
);

// Duplicate to NW and SE.
t!(
    t_007d72e733d944bfbf5c024ad4ea1ff8,
    "max(|x|, |y|) = 7",
    @dilate("1,0,0;0,0,0;0,0,1"),
    @size(8, 8)
);

// Duplicate to NE and SW.
t!(
    t_b7ffc60468a74a70bd55201504f98801,
    "max(|x|, |y|) = 7",
    @dilate("0,0,1;0,0,0;1,0,0"),
    @size(8, 8)
);

// Asymmetric shape.
t!(
    t_f810bce2133b46ebb23fdd4c8bdc1d56,
    "x = 0.01 ∧ y = 0.01",
    @dilate("0,1,1,1,0;1,0,0,0,0;1,0,1,1,0;1,0,0,1,0;0,1,1,1,0"),
    @size(8, 8)
);

// True pixels overwrite uncertain ones.
t!(
    t_63c475689a5643b4a8bb55177504e8a4,
    "(y - x - 0.01) |y + x - 0.01| = 0",
    @dilate("1,1,1;1,1,1;1,1,1"),
    @size(8, 8)
);
