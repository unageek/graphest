// From https://www.wolframalpha.com/input/?i=parametric+planar+curves

// Archimedes' spiral (a = 1/2)
t!(
    t_8844cc6c192544249d545aa4ff93fdfc,
    "x = 1/2 t cos(t) && y = 1/2 t sin(t) && 0 ≤ t ≤ 6π"
);

// astroid (a = 5)
t!(
    t_886f8b99d8394e8eae76907506d95f1e,
    "x = 5 cos(t)^3 && y = 5 sin(t)^3"
);

// Atzema spiral (a = 1)
t!(
    t_812458dc9b594ace92306e2941a48539,
    "x = -t sin(t) + sin(t) / t - 2 cos(t) && y = -2 sin(t) + t cos(t) - cos(t) / t && 1 ≤ t ≤ 10"
);

// bifoliate (a = 5)
t!(
    t_98966651db534aa79c7e026b6d2171dd,
    "x = 40 sin(t)^2 cos(t)^2 / (cos(4 t) + 3) && y = 40 sin(t)^3 cos(t) / (cos(4 t) + 3)",
    "--timeout",
    "2000"
);

// bifolium (a = 5)
t!(
    t_af16b5074e804e86bc2bd7ce2fc07db0,
    "x = 20 sin(t)^2 cos(t)^2 && y = 20 sin(t)^3 cos(t)"
);

// second butterfly curve
t!(t_1fc142fa2f4b49bc993234a1384220b0, "x = sin(t) (sin(t / 12)^5 + exp(cos(t)) - 2 cos(4 t)) && y = cos(t) (sin(t / 12)^5 + exp(cos(t)) - 2 cos(4 t))", "--timeout", "2000");

// cardioid (a = 2)
t!(
    t_302532e32f2f46ea96c82f312543de97,
    "x = 2 (1 - cos(t)) cos(t) && y = 2 sin(t) (1 - cos(t))"
);

// Cayley sextic (a = 5)
t!(
    t_64942ef8f2d5401882f0d91820e506ac,
    "x = 5 cos(t / 3)^3 cos(t) && y = 5 sin(t) cos(t / 3)^3"
);

// circle (a = 5)
t!(
    t_6acfe3a561814938a3a3b01d5514e2ca,
    "x = 5 cos(t) && y = 5 sin(t)"
);

// circle involute (a = 1)
t!(
    t_8aa880fe2833467084836d7dd21198c9,
    "x = t sin(t) + cos(t) && y = sin(t) - t cos(t) && 0 ≤ t ≤ 10"
);

// cycloid (a = 1)
t!(
    t_f2c688e095a9469fa6483e0d3dd0e36a,
    "x = t - sin(t) && y = 1 - cos(t)"
);

// deltoid (a = 5)
t!(
    t_268ee2e1bec147d8989ab63963769bb0,
    "x = 5 (2 cos(t) / 3 + cos(2 t) / 3) && y = 5 (2 sin(t) / 3 - sin(2 t) / 3)"
);

// folium of Descartes (a = 5)
t!(
    t_f85bacb939644c6381c22568c3e8a54f,
    "x = 15 t / (t^3 + 1) && y = 15 t^2 / (t^3 + 1) && -30 ≤ t ≤ 30"
);

// cissoid of Diocles (a = 2)
t!(
    t_539949c082ff4b319c98c9f68e9a3d14,
    "x = 4 sin(t)^2 && y = 4 sin(t)^2 tan(t)"
);

// eight curve (a = 5)
t!(
    t_7c7632d36c2c49499bbb8ac454a0063a,
    "x = 5 sin(t) && y = 5 sin(t) cos(t)",
    "--timeout",
    "2000"
);

// Doppler spiral (a = 1/10, k = 2)
t!(
    t_55ac60d148b44bbe9aad186ee255c61c,
    "x = 1/10 (2 t + t cos(t)) && y = 1/10 t sin(t) && 0 ≤ t ≤ 40"
);

// Freeth nephroid (a = 2)
t!(
    t_dca178704b6b4f259552f104208e8110,
    "x = 2 (2 sin(t / 2) + 1) cos(t) && y = 2 (2 sin(t / 2) + 1) sin(t)"
);

// Garfield curve (a = 1)
t!(
    t_9d71fb2ea4844f699a78bebc123772e2,
    "x = t cos(t)^2 && y = t sin(t) cos(t) && -2π ≤ t ≤ 2π"
);

// fourth heart curve (a = 2)
t!(
    t_b5438096e28242eea92db15e8786a563,
    "x = 2 cos(t) (sin(t) sqrt(|cos(t)|) / (sin(t) + 7/5) - 2 sin(t) + 2) && y = 2 sin(t) (sin(t) sqrt(|cos(t)|) / (sin(t) + 7/5) - 2 sin(t) + 2)"
);

// fifth heart curve
t!(
    t_0e2ef4e21ed34beeb4279b4c5c36a228,
    "x = 16 sin(t)^3 && y = 13 cos(t) - 5 cos(2 t) - 2 cos(3 t) - cos(4 t)"
);

// parabola involute (a = 2)
t!(
    t_14c9b1bb6ad04e4bb1c5252a73c96e51,
    "x = -1/2 t tanh(t) && y = 1/2 (sinh(t) - t / cosh(t))"
);

// piriform curve (a = 5, b = 5)
t!(
    t_1e66c48b91f74a9592b2c53adc73bb37,
    "x = 5 sin(t) + 1 && y = 5 (sin(t) + 1) cos(t)"
);

// ranunculoid (a = 1)
t!(
    t_4f1d55a0923c4d4fb6cbcc67ea8b12fb,
    "x = 6 cos(t) - cos(6 t) && y = 6 sin(t) - sin(6 t)"
);

// tractrix (a = 2)
t!(
    t_953663f505aa46f2a7dfbb460e5da84a,
    "x = 2 (t - tanh(t)) && y = 2 / cosh(t)"
);

// tractrix spiral (a = 10)
t!(
    t_ec7bc6583a004fca8a4dfefe5d24ca8b,
    "x = 10 cos(t) cos(t - tan(t)) && y = 10 cos(t) sin(t - tan(t)) && 0 ≤ t < π/2",
    "--timeout",
    "2000"
);

// Tschirnhausen cubic (a = 1/10)
t!(
    t_7da3ecab3414467aa8d3e832e231e70c,
    "x = 1/10 (1 - 3 t^2) && y = 1/10 t(3 - t^2)"
);
