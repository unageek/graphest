// Non-square
t!(t_b0561718fd3541d2958de31f0310e101, "y = sin(exp(x))", @size(456, 789));
t!(t_900f23b5cd764428b608611b30859d6e, "y = sin(exp(x))", @size(789, 456));
t!(t_8f83795722ca41349fe0e31fa80447f9, "x = sin(exp(y))", @size(456, 789));
t!(t_bc85afb687af42daaedfa0db21afb2a1, "x = sin(exp(y))", @size(789, 456));

// Inequality
t!(t_ef468eca14b4420fbcd9d886ca6a6651, "y < 5 sgn(x)");
t!(t_db103391e8bf44b78f4fb84de7813765, "y ≤ 5 sgn(x)");
t!(t_79fe1f66c9a447fbbdd34c9b19ed1a14, "y > 5 sgn(x)");
t!(t_e5a219e98542419cbdbe128ff1a88c8a, "y ≥ 5 sgn(x)");
