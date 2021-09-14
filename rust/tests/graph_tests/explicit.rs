// Non-square
t!(t_b0561718fd3541d2958de31f0310e101, "y = sin(exp(x))", @size(456, 789));
t!(t_900f23b5cd764428b608611b30859d6e, "y = sin(exp(x))", @size(789, 456));
t!(t_8f83795722ca41349fe0e31fa80447f9, "x = sin(exp(y))", @size(456, 789));
t!(t_bc85afb687af42daaedfa0db21afb2a1, "x = sin(exp(y))", @size(789, 456));

// Non-centered
t!(t_ddbf2d9ea2644e10a84f23c58c156870, "y = sin(exp(x))", @bounds(-12, 8, -4, 16));
t!(t_da2f53e4fdc9456a907e308edfbddf4b, "x = sin(exp(y))", @bounds(-4, 16, -12, 8));

// Inequality
t!(t_12ec2e045b3c497bb3e16a73dd07e7d6, "y < sqrt(x)");
t!(t_efde3acd852446e689c48f1490506d24, "y ≤ sqrt(x)");
t!(t_db32206574674f008c1db254e97f2742, "y > sqrt(x)");
t!(t_c8bcbe5b1ca849a9bf413518c22d4fa4, "y ≥ sqrt(x)");
