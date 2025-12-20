// From https://www.wolframalpha.com/input/?i=polar+curves

// bifoliate (a = 1)
t!(
    t_b1b247d3dba94a98a1e89ae9099bb442,
    "r = 8 sin(θ)^2 cos(θ) / (cos(4 θ) + 3)",
    @bounds(-2, 2, -2, 2),
    @timeout(6000),
);

// bifolium (a = 1)
t!(
    t_cc34e20f7186486690b8cf44e6eee27f,
    "r = 4 sin(θ)^2 cos(θ)",
    @bounds(-2, 2, -2, 2),
    @timeout(3000),
);

// cardioid (a = 1)
t!(
    t_4ccd608823904d1a935a15cc36d17c7f,
    "r = 1 - cos(θ)",
    @bounds(-5, 5, -5, 5),
);

// Cayley sextic (a = 1)
t!(
    t_a96f22e561114a4ca946e8e0aae6d124,
    "r = cos(θ / 3)^3",
    @bounds(-2, 2, -2, 2),
    @timeout(3000),
);

// cycloid of Ceva (a = 1)
t!(
    t_cd74d6c729494e8a9249b739d8986e50,
    "r = 2 cos(2 θ) + 1",
    @bounds(-5, 5, -5, 5),
);

// circle (a = 1)
t!(
    t_aa74fbf9a5ae4501a4e5bd9730a75723,
    "r = 1",
    @bounds(-2, 2, -2, 2),
);

// kampyle of Eudoxus (a = 1)
t!(t_b6ca9a802ce4498f8b8b5e92c30e1fa6, "r = 1 / cos(θ)^2");

// Freeth nephroid (a = 1)
t!(
    t_53a9535dc7a348c6b8eea80e7a27982f,
    "r = 2 sin(θ / 2) + 1",
    @bounds(-5, 5, -5, 5),
);

// Garfield curve (a = 1)
t!(
    t_34429ea795494b0593297fdb67e1c823,
    "r = θ cos(θ) ∧ -2π ≤ θ ≤ 2π",
    @timeout(2000),
);

// fourth heart curve (a = 1)
t!(
    t_d32ed36d48f94fc8960be7e237e99913,
    "r = sin(θ) sqrt(|cos(θ)|) / (sin(θ) + 7/5) - 2 sin(θ) + 2",
    @bounds(-5, 5, -5, 5),
    @timeout(2000),
);

// lituus (a = 1)
t!(
    t_6c81a6bd650d4af2806953aa6668fc7f,
    "r = 1 / sqrt(θ)",
    @bounds(-1, 1, -1, 1),
    @timeout(2000),
);

// Maltese cross curve (a = 1)
t!(t_4e807a8c807c419e8b1442aced86251f, "r = 2 / sqrt(sin(4 θ))");

// trifolium (a = 1)
t!(
    t_ec54de81941547d9a1bfec6f09badf75,
    "r = -cos(3θ)",
    @bounds(-2, 2, -2, 2),
    @timeout(2000),
);

// Tschirnhausen cubic (a = 1)
t!(
    t_1a810505ca14423bbe7d14eb1d03119c,
    "r = 1 / cos(θ / 3)^3",
    @bounds(-100, 100, -100, 100),
);

// Others

t!(
    t_7f564d75d3de45faa732ffdcec3f7ea5,
    "θ = 2π n/12 ∧ 0 ≤ n < 12",
);
