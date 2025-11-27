t!(t_9141185741e3473bafdd4fbd69daacb0, "y = abs(x)");
t!(t_3424e84276854880bfbaf722a55f4083, "y = acos(x)");
t!(t_37f5ce66894a4b94a578df440066cf19, "y = acosh(x)");
t!(t_38eec1c4b37442ce82c57cca99f36e81, "y = Ai(x)");
t!(t_df6c0b5de0cb45beb1dfe37e25b5c3ba, "y = Ai'(x)");
t!(t_e46074d3fb3d4876aa7cde273ded0a34, "y = asin(x)");
t!(t_9301c82b5f894f16b9b5122c09221aee, "y = asinh(x)");
t!(t_0444c061f4974171a76c9726967ecdd4, "y = atan(x)");
t!(t_d0cb560e76f64a2c96b51cc659192105, "y = atanh(x)");
t!(t_0dd58ad0988349c3bf70719d5db1b195, "y = Bi(x)");
t!(t_99d818f7f192402a9c7579b1e3bda62b, "y = Bi'(x)");
t!(
    t_a5daa83f3ba6452694b8dc31989e49a5,
    "y = C(x)",
    @timeout(4000),
);
t!(t_e2432ca1dca74defb99c089a7d232eeb, "y = ceil(x)");
t!(t_3c8809b76d304630b6bb047ed44c3c84, "y = Chi(x)");
t!(t_2f00a24bfa8c4d4ab2f74382299d1029, "y = Ci(x)");
t!(t_4e6cef6277de458aa116679f52721b31, "y = cos(x)");
t!(t_ed14cf332fd64c8398d072b14dd93d54, "y = cosh(x)");
t!(
    t_627f51aa56d14309b527f2a37b6368e5,
    "y = E(x)",
    @timeout(2000),
);
t!(t_db337020ebbb4d4f82d930d38c09ee7c, "y = Ei(x)");
t!(t_ae7ed0eb236e4e579b04aadf26a2d767, "y = erf(x)");
t!(t_93156a90660c43ce987d0cab7316cba3, "y = erfc(x)");
t!(t_1cc0b236a5c34d2a81bc5dc6f3152fcf, "y = erfcinv(x)");
t!(t_99de97a06c824edfbccd3e425633a290, "y = erfi(x)");
t!(t_41c51c4052894908ad252aa1f45f30ac, "y = erfinv(x)");
t!(t_944c89064a004af9bcda883f98785d4c, "y = exp(x)");
t!(t_0d3db3fb75604b85893ef6935b6f6ce8, "y = floor(x)");
t!(t_4e62c6b2dc704f47a8275b5f412b6f6c, "y = Γ(x)");
t!(t_50f32698582b4c4da21202d71998a54b, "y = lnΓ(x)");
t!(t_9903294009524c4aa671b035804cd213, "y = Γ(-1, x)");
t!(t_7ba9b50587f64182a5fb7130232843f6, "y = Γ(0, x)");
t!(t_3d165e6f280540cf88da4b1ececeb299, "y = Γ(1, x)");
t!(t_eafdf80bfc1d45f6a244dfa827b389f9, "y = Γ(2, x)");
t!(t_4f6d9c9c7b8d484993cb265a88322897, "y = K(x)");
t!(t_a1781b46eb8c45f58d656f455677c84c, "y = li(x)");
t!(t_6c486f3c2aef49dead6acd55f3848433, "y = ln(x)");
t!(t_09177d1cf2fb44af91b5e47a7213019d, "y = log(10, x)");
t!(t_0e998089005f4afda4752098d63cd55a, "y = ψ(x)");
t!(
    t_a99a60361a8d4b69a1461484c0622e10,
    "y = S(x)",
    @timeout(4000),
);
t!(t_15e48b4aa9864b56bc295596008ced2f, "y = sgn(x)");
t!(t_c6e28ae9902e4517885c39c5c776e26a, "y = Shi(x)");
t!(t_1a93bd43905e4d7eb041e4fe75175534, "y = Si(x)");
t!(t_b9fb18d5692e4ae6ab6e011ee31e32e6, "y = sin(x)");
t!(t_2167a4da30594e209d49215d180c6edb, "y = sinc(x)");
t!(t_5ac006668e084891a8394362499446ea, "y = sinh(x)");
t!(t_58ee894ce5fb4164ac0e3c422ac7948f, "y = sqrt(x)");
t!(t_51f6423ada6a45eba368eb03b57e10ee, "y = tan(x)");
t!(t_c88349f5b4a74502b82b56c56ffd9ea7, "y = tanh(x)");
t!(t_0a655aecc017422a8a829bb33f8ff280, "y = W(x)");
t!(t_91b4b44592264430835999d3ec1b0363, "y = W(-1, x)");
t!(t_e2a2a39070e6465fb4ce10c6b6ab3588, "y = ζ(x)");

// Bessel functions
t!(
    t_1766086ff8d848829170cead2dc18e7b,
    "y = J(-3/2, x)",
    @bounds(-4, 4, -4, 4),
);
t!(
    t_4f55a2d3f14b4539b5a1c38277a1aaf1,
    "y = J(-1, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(3000),
);
t!(
    t_e657f897db644391905f6245dfd67df1,
    "y = J(-1/2, x)",
    @bounds(-4, 4, -4, 4),
);
t!(
    t_f8780e7966644953b23ded3a6ce8e2a6,
    "y = J(0, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(4000),
);
t!(
    t_b7e50ea47abe42559b286b66964498d7,
    "y = J(1/2, x)",
    @bounds(-4, 4, -4, 4),
);
t!(
    t_6334b67231b84fcaa62320fa65b7c77f,
    "y = J(1, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(3000),
);
t!(
    t_49dd5ac993f24e078ae1a0606b95a49f,
    "y = J(3/2, x)",
    @bounds(-4, 4, -4, 4),
);

t!(
    t_19532c1221be449481385eaf7c42a279,
    "y = Y(-3/2, x)",
    @bounds(-4, 4, -4, 4),
);
t!(
    t_2bba64e6e7d24423bbd3d2a08fe5c780,
    "y = Y(-1, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(5000),
);
t!(
    t_1d96ba0ba09f44e7a9e3f497526013fa,
    "y = Y(-1/2, x)",
    @bounds(-4, 4, -4, 4),
);
t!(
    t_9359b338708e4df1a8bdc6220b95e527,
    "y = Y(0, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(3000),
);
t!(
    t_795e8b2422db48ca86b5de4a2de94b4c,
    "y = Y(1/2, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(3000),
);
t!(
    t_a95ce899a2c24239b21bd1f787610d4b,
    "y = Y(1, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(5000),
);
t!(
    t_7432071d9b3c4526bfe61c05f91e1964,
    "y = Y(3/2, x)",
    @bounds(-4, 4, -4, 4),
    @timeout(2000),
);

t!(t_27eebd1d353b4aada53c8dfd22f7e08c, "y = I(-3/2, x)",);
t!(t_eaa5ae3d73ac4599a0dcb865278684f2, "y = I(-1, x)",);
t!(t_a495cd2e3ac141729ce70da2bbf69005, "y = I(-1/2, x)",);
t!(t_8355dc4b93bf4a8896db89c7abdb804f, "y = I(0, x)",);
t!(t_6a5ff0062f8c46ad9bbb4c8dc7d9681c, "y = I(1/2, x)",);
t!(t_961ac5f68d8747db8b1e701cd0138931, "y = I(1, x)",);
t!(t_6a99a6054e314aedb19af97d03ca66f8, "y = I(3/2, x)",);

t!(t_0c5785ed91294fd9b87815926fb92c4a, "y = K(-3/2, x)",);
t!(t_24fee629fc8d49008760b5da7a2122c6, "y = K(-1, x)",);
t!(t_17e51ff3f84d463384561a781a909c5c, "y = K(-1/2, x)",);
t!(t_52c7fee2d49a4cc7ab8500dc20bc917d, "y = K(0, x)",);
t!(t_d90cfe5768e045d2b2f06f6a17de8f45, "y = K(1/2, x)",);
t!(t_2a7837af7fa54a4ca9da271ac72f339b, "y = K(1, x)",);
t!(t_b81dbc5c9ffa4286922c2989ef730638, "y = K(3/2, x)",);
