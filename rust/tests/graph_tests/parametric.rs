// Non-square
t!(
    t_90e85ab048f04560bfc9eef9e32c4b9c,
    "x = 8 cos(t) ∧ y = 8 sin(t) ∧ 1 < t < 5",
    @size(456, 789)
);
t!(
    t_a039ce4efca04d99b6a6bdd08b5b2bf4,
    "x = 8 cos(t) ∧ y = 8 sin(t) ∧ 1 < t < 5",
    @size(789, 456)
);

// From https://www.wolframalpha.com/input/?i=parametric+planar+curves

// Archimedes' spiral (a = 1)
t!(
    t_8844cc6c192544249d545aa4ff93fdfc,
    "x = t cos(t) ∧ y = t sin(t) ∧ 0 ≤ t ≤ 6π",
    @bounds(-20, 20, -20, 20),
);

// astroid (a = 1)
t!(
    t_886f8b99d8394e8eae76907506d95f1e,
    "x = cos(t)^3 ∧ y = sin(t)^3",
    @bounds(-2, 2, -2, 2),
);

// Atzema spiral (a = 1)
t!(
    t_812458dc9b594ace92306e2941a48539,
    "x = -t sin(t) + sin(t) / t - 2 cos(t) ∧ y = -2 sin(t) + t cos(t) - cos(t) / t ∧ 1 ≤ t ≤ 10",
);

// bifoliate (a = 1)
t!(
    t_98966651db534aa79c7e026b6d2171dd,
    "x = 8 sin(t)^2 cos(t)^2 / (cos(4 t) + 3) ∧ y = 8 sin(t)^3 cos(t) / (cos(4 t) + 3)",
    @bounds(-2, 2, -2, 2),
    @timeout(2000),
);

// bifolium (a = 1)
t!(
    t_af16b5074e804e86bc2bd7ce2fc07db0,
    "x = 4 sin(t)^2 cos(t)^2 ∧ y = 4 sin(t)^3 cos(t)",
    @bounds(-2, 2, -2, 2),
);

// second butterfly curve
t!(
    t_1fc142fa2f4b49bc993234a1384220b0,
    "x = sin(t) (sin(t / 12)^5 + exp(cos(t)) - 2 cos(4 t)) ∧ y = cos(t) (sin(t / 12)^5 + exp(cos(t)) - 2 cos(4 t))",
    @bounds(-5, 5, -5, 5),
    @timeout(4000),
);

// cardioid (a = 1)
t!(
    t_302532e32f2f46ea96c82f312543de97,
    "x = (1 - cos(t)) cos(t) ∧ y = sin(t) (1 - cos(t))",
    @bounds(-5, 5, -5, 5),
);

// Cayley sextic (a = 1)
t!(
    t_64942ef8f2d5401882f0d91820e506ac,
    "x = cos(t / 3)^3 cos(t) ∧ y = sin(t) cos(t / 3)^3",
    @bounds(-2, 2, -2, 2),
);

// cycloid of Ceva (a = 1)
t!(
    t_b4bf85744ce444fb83f38ca9c6c56b30,
    "x = cos(t) (2 cos(2 t) + 1) ∧ y = sin(t) (2 cos(2 t) + 1)",
    @bounds(-5, 5, -5, 5),
);

// circle (a = 1)
t!(
    t_6acfe3a561814938a3a3b01d5514e2ca,
    "x = cos(t) ∧ y = sin(t)",
    @bounds(-2, 2, -2, 2),
);

// circle involute (a = 1)
t!(
    t_8aa880fe2833467084836d7dd21198c9,
    "x = t sin(t) + cos(t) ∧ y = sin(t) - t cos(t) ∧ 0 ≤ t ≤ 10",
);

// cycloid (a = 1)
t!(
    t_f2c688e095a9469fa6483e0d3dd0e36a,
    "x = t - sin(t) ∧ y = 1 - cos(t)",
);

// deltoid (a = 1)
t!(
    t_268ee2e1bec147d8989ab63963769bb0,
    "x = 2 cos(t) / 3 + cos(2 t) / 3 ∧ y = 2 sin(t) / 3 - sin(2 t) / 3",
    @bounds(-2, 2, -2, 2),
);

// folium of Descartes (a = 1)
t!(
    t_f85bacb939644c6381c22568c3e8a54f,
    "x = 3 t / (t^3 + 1) ∧ y = 3 t^2 / (t^3 + 1) ∧ -30 ≤ t ≤ 30",
    @bounds(-2, 2, -2, 2),
);

// cissoid of Diocles (a = 1)
t!(
    t_539949c082ff4b319c98c9f68e9a3d14,
    "x = 2 sin(t)^2 ∧ y = 2 sin(t)^2 tan(t)",
    @bounds(-5, 5, -5, 5),
);

// Doppler spiral (a = 1, k = 2)
t!(
    t_55ac60d148b44bbe9aad186ee255c61c,
    "x = 2 t + t cos(t) ∧ y = t sin(t) ∧ 0 ≤ t ≤ 40",
    @bounds(-50, 150, -100, 100),
);

// eight curve (a = 1)
t!(
    t_7c7632d36c2c49499bbb8ac454a0063a,
    "x = sin(t) ∧ y = sin(t) cos(t)",
    @bounds(-2, 2, -2, 2),
    @timeout(2000),
);

// kampyle of Eudoxus (a = 1)
t!(
    t_83353dd50cd441c2b99967cd1333a601,
    "x = 1 / cos(t) ∧ y = tan(t) / cos(t)",
);

// Freeth nephroid (a = 1)
t!(
    t_dca178704b6b4f259552f104208e8110,
    "x = (2 sin(t / 2) + 1) cos(t) ∧ y = (2 sin(t / 2) + 1) sin(t)",
    @bounds(-5, 5, -5, 5),
);

// Garfield curve (a = 1)
t!(
    t_9d71fb2ea4844f699a78bebc123772e2,
    "x = t cos(t)^2 ∧ y = t sin(t) cos(t) ∧ -2π ≤ t ≤ 2π",
);

// fourth heart curve (a = 1)
t!(
    t_b5438096e28242eea92db15e8786a563,
    "x = cos(t) (sin(t) sqrt(|cos(t)|) / (sin(t) + 7/5) - 2 sin(t) + 2) ∧ y = sin(t) (sin(t) sqrt(|cos(t)|) / (sin(t) + 7/5) - 2 sin(t) + 2)",
    @bounds(-5, 5, -5, 5),
);

// fifth heart curve
t!(
    t_0e2ef4e21ed34beeb4279b4c5c36a228,
    "x = 16 sin(t)^3 ∧ y = 13 cos(t) - 5 cos(2 t) - 2 cos(3 t) - cos(4 t)",
    @bounds(-20, 20, -20, 20),
);

// lituus (a = 1)
t!(
    t_c6f55e36d9ce4673bad541e0b1ad5f24,
    "x = cos(t) / sqrt(t) ∧ y = sin(t) / sqrt(t)",
    @bounds(-1, 1, -1, 1),
    @timeout(5000),
);

// Maltese cross curve (a = 1)
t!(
    t_e80f0dbadfeb49248226ec4cc859385f,
    "x = 2 cos(t) / sqrt(sin(4 t)) ∧ y = 2 sin(t) / sqrt(sin(4 t))",
);

// parabola involute (a = 1)
t!(
    t_14c9b1bb6ad04e4bb1c5252a73c96e51,
    "x = -1/4 t tanh(t) ∧ y = 1/4 (sinh(t) - t / cosh(t))",
    @bounds(-5, 5, -5, 5),
);

// piriform curve (a = 1, b = 1)
t!(
    t_1e66c48b91f74a9592b2c53adc73bb37,
    "x = sin(t) + 1 ∧ y = (sin(t) + 1) cos(t)",
    @bounds(-1, 3, -2, 2),
);

// ranunculoid (a = 1)
t!(
    t_4f1d55a0923c4d4fb6cbcc67ea8b12fb,
    "x = 6 cos(t) - cos(6 t) ∧ y = 6 sin(t) - sin(6 t)",
);

// tractrix (a = 1)
t!(
    t_953663f505aa46f2a7dfbb460e5da84a,
    "x = t - tanh(t) ∧ y = 1 / cosh(t)",
    @bounds(-5, 5, -5, 5),
);

// tractrix spiral (a = 1)
t!(
    t_ec7bc6583a004fca8a4dfefe5d24ca8b,
    "x = cos(t) cos(t - tan(t)) ∧ y = cos(t) sin(t - tan(t)) ∧ 0 ≤ t < π/2",
    @bounds(-1, 1, -1, 1),
    @timeout(3000),
);

// trifolium (a = 1)
t!(
    t_2ce09882cb1a4de9a007f124c2dc36af,
    "x = -cos(t) cos(3 t) ∧ y = -sin(t) cos(3 t)",
    @bounds(-2, 2, -2, 2),
);

// Tschirnhausen cubic (a = 1)
t!(
    t_7da3ecab3414467aa8d3e832e231e70c,
    "x = 1 - 3 t^2 ∧ y = t (3 - t^2)",
    @bounds(-100, 100, -100, 100),
);

// Integer parameters

t!(
    t_edb96325adb6497cbb0173eb9d5496cc,
    "x = 10 / m ∧ y = 20 / n",
);

t!(
    t_c6947324ad53422e9076ce55e9073913,
    "x = 10 / m ∧ y = 20 / n ∧ m = n",
);

t!(
    t_bfe7a22400304fa0bf9d0260f522f047,
    "x = cos(t) + m ∧ y = sin(t) + 2 n ∧ m = n",
);
