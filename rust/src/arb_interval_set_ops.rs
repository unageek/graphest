use crate::{
    interval_set::{Site, TupperInterval, TupperIntervalSet},
    interval_set_ops,
};
use inari::{const_interval, interval, DecInterval, Decoration, Interval};
use itertools::Itertools;
use std::ops::{BitAnd, BitOr};

#[derive(Clone, Copy, Eq, Debug, PartialEq)]
struct BoolInterval(bool, bool);

impl BoolInterval {
    pub const TRUE: Self = Self(true, true);

    pub fn new(a: bool, b: bool) -> Self {
        assert!(!a || b); // a → b
        Self(a, b)
    }

    pub fn certainly(self) -> bool {
        self.0
    }

    pub fn possibly(self) -> bool {
        self.1
    }
}

macro_rules! ge {
    ($x:expr, $y:expr) => {{
        static_assertions::const_assert!(f64::NEG_INFINITY < $y && $y < f64::INFINITY);
        BoolInterval::new($x.inf() >= $y, $x.sup() >= $y)
    }};
}

macro_rules! gt {
    ($x:expr, $y:expr) => {{
        static_assertions::const_assert!(f64::NEG_INFINITY < $y && $y < f64::INFINITY);
        BoolInterval::new($x.inf() > $y, $x.sup() > $y)
    }};
}

macro_rules! le {
    ($x:expr, $y:expr) => {{
        static_assertions::const_assert!(f64::NEG_INFINITY < $y && $y < f64::INFINITY);
        BoolInterval::new($x.sup() <= $y, $x.inf() <= $y)
    }};
}

macro_rules! lt {
    ($x:expr, $y:expr) => {{
        static_assertions::const_assert!(f64::NEG_INFINITY < $y && $y < f64::INFINITY);
        BoolInterval::new($x.sup() < $y, $x.inf() < $y)
    }};
}

macro_rules! ne {
    ($x:expr, $y:expr) => {{
        static_assertions::const_assert!(f64::NEG_INFINITY < $y && $y < f64::INFINITY);
        BoolInterval::new(!$x.contains($y), $x != const_interval!($y, $y))
    }};
}

impl BitAnd for BoolInterval {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::new(self.0 && rhs.0, self.1 && rhs.1)
    }
}

impl BitOr for BoolInterval {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::new(self.0 || rhs.0, self.1 || rhs.1)
    }
}

macro_rules! impl_arb_op {
    ($op:ident($x:ident), $result:expr) => {
        impl_arb_op!($op($x), $result, BoolInterval::TRUE);
    };

    ($op:ident($x:ident), $result:expr, $def:expr) => {
        pub fn $op(&self) -> Self {
            let mut rs = Self::new();
            for x in self {
                let $x = x.x;
                let def = $def;
                if def.possibly() {
                    let dec = if def.certainly() {
                        // Assuming the restriction of f to x is continuous.
                        Decoration::Dac.min(x.d)
                    } else {
                        Decoration::Trv
                    };
                    rs.insert(TupperInterval::new(DecInterval::set_dec($result, dec), x.g));
                }
            }
            rs.normalize(false);
            rs
        }
    };

    ($op:ident($x:ident, $y:ident), $result:expr, $def:expr) => {
        pub fn $op(&self, rhs: &Self) -> Self {
            let mut rs = Self::new();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.x;
                        let $y = y.x;
                        let def = $def;
                        if def.possibly() {
                            let dec = if def.certainly() {
                                // Assuming the restriction of f to x × y is continuous.
                                Decoration::Dac.min(x.d).min(y.d)
                            } else {
                                Decoration::Trv
                            };
                            rs.insert(TupperInterval::new(DecInterval::set_dec($result, dec), g));
                        }
                    }
                }
            }
            rs.normalize(false);
            rs
        }
    };
}

fn i(x: f64) -> Interval {
    interval!(x, x).unwrap()
}

const M_ONE_TO_ONE: Interval = const_interval!(-1.0, 1.0);
const N_INF_TO_ZERO: Interval = const_interval!(f64::NEG_INFINITY, 0.0);
const ONE_HALF: Interval = const_interval!(0.5, 0.5);
const ONE_TO_INF: Interval = const_interval!(1.0, f64::INFINITY);
const ZERO: Interval = const_interval!(0.0, 0.0);
const ZERO_TO_INF: Interval = const_interval!(0.0, f64::INFINITY);
const ZERO_TO_ONE: Interval = const_interval!(0.0, 1.0);

impl TupperIntervalSet {
    // Mid-rad IA, which is used by Arb, cannot represent half-bounded intervals.
    // So we need to handle such inputs and unbounded functions explicitly.

    impl_arb_op!(
        acos(x),
        if x.interior(M_ONE_TO_ONE) {
            arb_acos(x)
        } else {
            x.acos()
        },
        ge!(x, -1.0) & le!(x, 1.0)
    );

    impl_arb_op!(
        acosh(x),
        if x.inf() > 1.0 && x.sup() < f64::INFINITY {
            arb_acosh(x)
        } else {
            x.acosh()
        },
        ge!(x, 1.0)
    );

    impl_arb_op!(airy_ai(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [0, Ai(a)]
            interval!(0.0, arb_airy_ai(i(a)).sup()).unwrap()
        } else {
            arb_airy_ai(x).intersection(airy_envelope(x))
        }
    });

    impl_arb_op!(airy_ai_prime(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Ai'(a), 0]
            interval!(arb_airy_ai_prime(i(a)).inf(), 0.0).unwrap()
        } else {
            arb_airy_ai_prime(x)
        }
    });

    impl_arb_op!(airy_bi(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Bi(a), +∞]
            interval!(arb_airy_bi(i(a)).inf(), f64::INFINITY).unwrap()
        } else {
            arb_airy_bi(x).intersection(airy_envelope(x))
        }
    });

    impl_arb_op!(airy_bi_prime(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Bi'(a), +∞]
            interval!(arb_airy_bi_prime(i(a)).inf(), f64::INFINITY).unwrap()
        } else {
            arb_airy_bi_prime(x)
        }
    });

    impl_arb_op!(
        asin(x),
        if x.interior(M_ONE_TO_ONE) {
            arb_asin(x)
        } else {
            x.asin()
        },
        ge!(x, -1.0) & le!(x, 1.0)
    );

    impl_arb_op!(
        asinh(x),
        if x.is_common_interval() {
            arb_asinh(x)
        } else {
            x.asinh()
        }
    );

    impl_arb_op!(
        atan(x),
        if x.is_common_interval() {
            arb_atan(x)
        } else {
            x.atan()
        }
    );

    pub fn atan2(&self, rhs: &Self, site: Option<Site>) -> Self {
        if self.iter().all(|x| x.x.is_common_interval())
            && rhs.iter().all(|x| x.x.is_common_interval())
            && self
                .iter()
                .cartesian_product(rhs.iter())
                .filter(|(x, y)| x.g.union(y.g).is_some())
                .all(|(y, x)| x.x.inf() > 0.0 || !y.x.contains(0.0))
        {
            let mut rs = Self::new();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let (y, x) = (x, y);
                        let dec = Decoration::Dac.min(x.d).min(y.d);
                        let z = arb_atan2(y.x, x.x);
                        rs.insert(TupperInterval::new(DecInterval::set_dec(z, dec), g));
                    }
                }
            }
            rs.normalize(false);
            rs
        } else {
            self.atan2_impl(rhs, site)
        }
    }

    impl_arb_op!(
        atanh(x),
        if x.interior(M_ONE_TO_ONE) {
            arb_atanh(x)
        } else {
            x.atanh()
        },
        gt!(x, -1.0) & lt!(x, 1.0)
    );

    impl_arb_op!(
        bessel_i(n, x),
        {
            if n.inf() % 2.0 == 0.0 {
                // n ∈ 2ℤ
                let x = x.abs();
                let a = x.inf();
                let b = x.sup();
                let inf = arb_bessel_i(n, i(a)).inf();
                let sup = if b == f64::INFINITY {
                    b
                } else {
                    arb_bessel_i(n, i(b)).sup()
                };
                interval!(inf, sup).unwrap()
            } else if n.inf() % 1.0 == 0.0 {
                // n ∈ 2ℤ + 1
                let a = x.inf();
                let b = x.sup();
                let inf = if a == f64::NEG_INFINITY {
                    a
                } else {
                    arb_bessel_i(n, i(a)).inf()
                };
                let sup = if b == f64::INFINITY {
                    b
                } else {
                    arb_bessel_i(n, i(b)).sup()
                };
                interval!(inf, sup).unwrap()
            } else if n.inf() > 0.0 {
                // n ∈ (0, +∞) ∖ ℤ
                let x = x.intersection(ZERO_TO_INF);
                if x.is_empty() || x == ZERO {
                    Interval::EMPTY
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let inf = if a == 0.0 {
                        0.0
                    } else {
                        arb_bessel_i(n, i(a)).inf()
                    };
                    let sup = if b == f64::INFINITY {
                        b
                    } else {
                        arb_bessel_i(n, i(b)).sup()
                    };
                    interval!(inf, sup).unwrap()
                }
            } else if n.inf() % 2.0 > -1.0 {
                // n ∈ (-1, 0) ∪ (-3, -2) ∪ …
                let y0 = {
                    let x = x.intersection(const_interval!(0.0, 0.5));
                    if x.is_empty() || x == ZERO {
                        Interval::EMPTY
                    } else {
                        let a = x.inf();
                        let b = x.sup();
                        interval!(arb_bessel_i(n, i(b)).inf(), arb_bessel_i(n, i(a)).sup()).unwrap()
                    }
                };
                let y1 = {
                    let x = x.intersection(const_interval!(0.5, f64::INFINITY));
                    if x.is_empty() {
                        Interval::EMPTY
                    } else {
                        arb_bessel_i(n, x)
                    }
                };
                y0.convex_hull(y1)
            } else {
                // n ∈ (-2, -1) ∪ (-4, -3) ∪ …
                let x = x.intersection(ZERO_TO_INF);
                if x.is_empty() || x == ZERO {
                    Interval::EMPTY
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let inf = arb_bessel_i(n, i(a)).inf();
                    let sup = if b == f64::INFINITY {
                        b
                    } else {
                        arb_bessel_i(n, i(b)).sup()
                    };
                    interval!(inf, sup).unwrap()
                }
            }
        },
        {
            assert!(
                n.is_singleton() && n.inf() % 0.5 == 0.0,
                "`I(n, x)` only permits integers and half-integers for `n`"
            );
            if n.inf() % 1.0 == 0.0 {
                BoolInterval::TRUE
            } else {
                gt!(x, 0.0)
            }
        }
    );

    impl_arb_op!(
        bessel_j(n, x),
        {
            if n.inf() % 1.0 == 0.0 {
                // n ∈ ℤ
                arb_bessel_j(n, x).intersection(bessel_envelope(n, x))
            } else {
                // Bisection at 1 is only valid for integer/half-integer orders.
                // The first extremum point can get arbitrarily close to the origin in a general order.
                // The same note applies to `bessel_y` and `bessel_i`.
                let y0 = {
                    let x = x.intersection(ZERO_TO_ONE);
                    if x.is_empty() || x == ZERO {
                        Interval::EMPTY
                    } else {
                        let a = x.inf();
                        let b = x.sup();
                        if n.inf() > 0.0 {
                            // n ∈ (0, +∞) ∖ ℤ
                            let inf = if a == 0.0 {
                                0.0
                            } else {
                                arb_bessel_j(n, i(a)).inf()
                            };
                            interval!(inf, arb_bessel_j(n, i(b)).sup()).unwrap()
                        } else if n.inf() % 2.0 > -1.0 {
                            // n ∈ (-1, 0) ∪ (-3, -2) ∪ …
                            interval!(arb_bessel_j(n, i(b)).inf(), arb_bessel_j(n, i(a)).sup())
                                .unwrap()
                        } else {
                            // n = (-2, -1) ∪ (-4, -3) ∪ …
                            interval!(arb_bessel_j(n, i(a)).inf(), arb_bessel_j(n, i(b)).sup())
                                .unwrap()
                        }
                    }
                };
                let y1 = {
                    let x = x.intersection(ONE_TO_INF);
                    if x.is_empty() {
                        Interval::EMPTY
                    } else {
                        arb_bessel_j(n, x).intersection(bessel_envelope(n, x))
                    }
                };
                y0.convex_hull(y1)
            }
        },
        {
            assert!(
                n.is_singleton() && n.inf() % 0.5 == 0.0,
                "`J(n, x)` only permits integers and half-integers for `n`"
            );
            if n.inf() % 1.0 == 0.0 {
                BoolInterval::TRUE
            } else {
                gt!(x, 0.0)
            }
        }
    );

    impl_arb_op!(
        bessel_k(n, x),
        {
            let x = x.intersection(ZERO_TO_INF);
            let a = x.inf();
            let b = x.sup();
            let inf = if b == f64::INFINITY {
                0.0
            } else {
                arb_bessel_k(n, i(b)).inf()
            };
            let sup = arb_bessel_k(n, i(a)).sup();
            interval!(inf, sup).unwrap()
        },
        {
            assert!(
                n.is_singleton() && n.inf() % 0.5 == 0.0,
                "`K(n, x)` only permits integers and half-integers for `n`"
            );
            gt!(x, 0.0)
        }
    );

    impl_arb_op!(
        bessel_y(n, x),
        {
            let y0 = {
                let x = x.intersection(ZERO_TO_ONE);
                if x.is_empty() || x == ZERO {
                    Interval::EMPTY
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let n_rem_2 = n.inf() % 2.0;
                    if n_rem_2 == -0.5 {
                        // n = -1/2, -5/2, …
                        let inf = if a == 0.0 {
                            0.0
                        } else {
                            arb_bessel_y(n, i(a)).inf()
                        };
                        interval!(inf, arb_bessel_y(n, i(b)).sup()).unwrap()
                    } else if n_rem_2 == -1.5 {
                        // n = -3/2, -7/2, …
                        let sup = if a == 0.0 {
                            0.0
                        } else {
                            arb_bessel_y(n, i(a)).sup()
                        };
                        interval!(arb_bessel_y(n, i(b)).inf(), sup).unwrap()
                    } else if n_rem_2 > -1.5 && n_rem_2 < -0.5 {
                        // n ∈ (-3/2, -1/2) ∪ (-7/2, -5/2) ∪ …
                        interval!(arb_bessel_y(n, i(b)).inf(), arb_bessel_y(n, i(a)).sup()).unwrap()
                    } else {
                        // n ∈ (-1/2, +∞) ∪ (-5/2, -3/2) ∪ (-9/2, -7/2) ∪ …
                        interval!(arb_bessel_y(n, i(a)).inf(), arb_bessel_y(n, i(b)).sup()).unwrap()
                    }
                }
            };
            let y1 = {
                let x = x.intersection(ONE_TO_INF);
                if x.is_empty() {
                    Interval::EMPTY
                } else {
                    arb_bessel_y(n, x).intersection(bessel_envelope(n, x))
                }
            };
            y0.convex_hull(y1)
        },
        {
            assert!(
                n.is_singleton() && n.inf() % 0.5 == 0.0,
                "`Y(n, x)` only permits integers and half-integers for `n`"
            );
            gt!(x, 0.0)
        }
    );

    impl_arb_op!(
        chi(x),
        {
            let x = x.intersection(ZERO_TO_INF);
            let a = x.inf();
            let b = x.sup();
            if x == ZERO_TO_INF {
                Interval::ENTIRE
            } else if a == 0.0 {
                // [-∞, Chi(b)]
                interval!(f64::NEG_INFINITY, arb_chi(i(b)).sup()).unwrap()
            } else if b == f64::INFINITY {
                // [Chi(a), +∞]
                interval!(arb_chi(i(a)).inf(), f64::INFINITY).unwrap()
            } else {
                arb_chi(x)
            }
        },
        gt!(x, 0.0)
    );

    impl_arb_op!(
        ci(x),
        {
            let x = x.intersection(ZERO_TO_INF);
            let a = x.inf();
            let b = x.sup();
            if a == 0.0 && b <= Interval::FRAC_PI_2.inf() {
                // [-∞, Ci(b)]
                let sup = arb_ci(i(b)).sup();
                interval!(f64::NEG_INFINITY, sup).unwrap()
            } else {
                arb_ci(x).intersection(ci_envelope(x))
            }
        },
        gt!(x, 0.0)
    );

    impl_arb_op!(cos(x), arb_cos(x));

    impl_arb_op!(
        cosh(x),
        if x.is_common_interval() {
            arb_cosh(x)
        } else {
            x.cosh()
        }
    );

    impl_arb_op!(
        ei(x),
        {
            let a = x.inf();
            let b = x.sup();
            if b <= 0.0 {
                // [Ei(b), Ei(a)]
                // When b = 0, inf(arb_ei([b, b])) = inf([-∞, +∞]) = -∞.
                let inf = arb_ei(i(b)).inf();
                let sup = if a == f64::NEG_INFINITY {
                    0.0
                } else {
                    arb_ei(i(a)).sup()
                };
                interval!(inf, sup).unwrap()
            } else if a >= 0.0 {
                // [Ei(a), Ei(b)]
                let inf = arb_ei(i(a)).inf();
                let sup = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_ei(i(b)).sup()
                };
                interval!(inf, sup).unwrap()
            } else {
                // [-∞, max(Ei(a), Ei(b))]
                let sup0 = if a == f64::NEG_INFINITY {
                    0.0
                } else {
                    arb_ei(i(a)).sup()
                };
                let sup1 = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_ei(i(b)).sup()
                };
                interval!(f64::NEG_INFINITY, sup0.max(sup1)).unwrap()
            }
        },
        ne!(x, 0.0)
    );

    impl_arb_op!(
        elliptic_e(x),
        {
            let a = x.inf();
            let b = x.sup();
            if a == f64::NEG_INFINITY && b >= 1.0 {
                const_interval!(1.0, f64::INFINITY)
            } else if a == f64::NEG_INFINITY {
                interval!(arb_elliptic_e(i(b)).inf(), f64::INFINITY).unwrap()
            } else if b >= 1.0 {
                interval!(1.0, arb_elliptic_e(i(a)).sup()).unwrap()
            } else {
                arb_elliptic_e(x)
            }
        },
        le!(x, 1.0)
    );

    impl_arb_op!(
        elliptic_k(x),
        {
            let a = x.inf();
            let b = x.sup();
            if a == f64::NEG_INFINITY && b >= 1.0 {
                const_interval!(0.0, f64::INFINITY)
            } else if a == f64::NEG_INFINITY {
                interval!(0.0, arb_elliptic_k(i(b)).sup()).unwrap()
            } else if b >= 1.0 {
                interval!(arb_elliptic_k(i(a)).inf(), f64::INFINITY).unwrap()
            } else {
                arb_elliptic_k(x)
            }
        },
        lt!(x, 1.0)
    );

    impl_arb_op!(
        erf(x),
        if x.is_common_interval() {
            arb_erf(x)
        } else {
            interval_set_ops::erf(x)
        }
    );

    impl_arb_op!(
        erfc(x),
        if x.is_common_interval() {
            arb_erfc(x)
        } else {
            interval_set_ops::erfc(x)
        }
    );

    impl_arb_op!(erfi(x), {
        let a = x.inf();
        let b = x.sup();
        if x.is_entire() {
            x
        } else if a == f64::NEG_INFINITY {
            // [-∞, erfi(b)]
            interval!(f64::NEG_INFINITY, arb_erfi(i(b)).sup()).unwrap()
        } else if b == f64::INFINITY {
            // [erfi(a), +∞]
            interval!(arb_erfi(i(a)).inf(), f64::INFINITY).unwrap()
        } else {
            arb_erfi(x)
        }
    });

    impl_arb_op!(
        exp(x),
        if x.is_common_interval() {
            arb_exp(x)
        } else {
            x.exp()
        }
    );

    impl_arb_op!(
        exp10(x),
        if x.is_common_interval() {
            arb_exp10(x)
        } else {
            x.exp10()
        }
    );

    impl_arb_op!(
        exp2(x),
        if x.is_common_interval() {
            arb_exp2(x)
        } else {
            x.exp2()
        }
    );

    impl_arb_op!(fresnel_c(x), {
        let a = x.inf();
        let b = x.sup();
        if b <= 0.0 {
            arb_fresnel_c(x).intersection(fresnel_envelope_centered(-x) - ONE_HALF)
        } else if a >= 0.0 {
            arb_fresnel_c(x).intersection(fresnel_envelope_centered(x) + ONE_HALF)
        } else {
            arb_fresnel_c(x)
        }
    });

    impl_arb_op!(fresnel_s(x), {
        let a = x.inf();
        let b = x.sup();
        if b <= 0.0 {
            arb_fresnel_s(x).intersection(fresnel_envelope_centered(-x) - ONE_HALF)
        } else if a >= 0.0 {
            arb_fresnel_s(x).intersection(fresnel_envelope_centered(x) + ONE_HALF)
        } else {
            arb_fresnel_s(x)
        }
    });

    pub fn gamma(&self, site: Option<Site>) -> Self {
        // NSolve[{Gamma[x] == $MaxMachineNumber, 0 < x < 180}, x]
        const X_LIMIT: f64 = 171.0;
        if self.iter().all(|x| {
            let a = x.x.inf();
            let b = x.x.sup();
            b < 0.0 && a.ceil() > b.floor() || a > 0.0 && b < X_LIMIT
        }) {
            let mut rs = Self::new();
            for x in self {
                let dec = Decoration::Com.min(x.d);
                let y = arb_gamma(x.x);
                rs.insert(TupperInterval::new(DecInterval::set_dec(y, dec), x.g));
            }
            rs.normalize(false);
            rs
        } else {
            self.gamma_impl(site)
        }
    }

    impl_arb_op!(
        gamma_inc(s, x),
        if s.inf() % 2.0 == 1.0 {
            // n = 1, 3, …
            let a = x.inf();
            let b = x.sup();
            let inf = if b == f64::INFINITY {
                0.0
            } else {
                arb_gamma_inc(s, i(b)).inf()
            };
            let sup = if a == f64::NEG_INFINITY {
                f64::INFINITY
            } else {
                arb_gamma_inc(s, i(a)).sup()
            };
            interval!(inf, sup).unwrap()
        } else {
            // n ≠ 1, 3, …
            let y0 = if s.inf() > 0.0 && s.inf() % 2.0 == 0.0 {
                // n = 2, 4, …
                let x = x.intersection(N_INF_TO_ZERO);
                if x.is_empty() {
                    Interval::EMPTY
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let inf = if a == f64::NEG_INFINITY {
                        a
                    } else {
                        arb_gamma_inc(s, i(a)).inf()
                    };
                    interval!(inf, arb_gamma_inc(s, i(b)).sup()).unwrap()
                }
            } else {
                // n ≠ 1, 2, …
                Interval::EMPTY
            };
            let y1 = {
                let x = x.intersection(ZERO_TO_INF);
                if x.is_empty() || s.inf() <= 0.0 && x == ZERO {
                    Interval::EMPTY
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let inf = if b == f64::INFINITY {
                        0.0
                    } else {
                        arb_gamma_inc(s, i(b)).inf()
                    };
                    interval!(inf, arb_gamma_inc(s, i(a)).sup()).unwrap()
                }
            };
            y0.convex_hull(y1)
        },
        {
            assert!(
                s.is_singleton(),
                "`Gamma(a, x)` only permits exact numbers for `a`"
            );
            let s = s.inf();
            if s > 0.0 && s % 1.0 == 0.0 {
                BoolInterval::TRUE
            } else {
                gt!(x, 0.0)
            }
        }
    );

    impl_arb_op!(
        li(x),
        {
            let x = x.intersection(ZERO_TO_INF);
            let a = x.inf();
            let b = x.sup();
            if b <= 1.0 {
                // [li(b), li(a)]
                interval!(arb_li(i(b)).inf(), arb_li(i(a)).sup()).unwrap()
            } else if a >= 1.0 {
                // [li(a), li(b)]
                let inf = arb_li(i(a)).inf();
                let sup = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_li(i(b)).sup()
                };
                interval!(inf, sup).unwrap()
            } else {
                // [-∞, max(li(a), li(b))]
                let sup0 = arb_li(i(a)).sup();
                let sup1 = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_li(i(b)).sup()
                };
                interval!(f64::NEG_INFINITY, sup0.max(sup1)).unwrap()
            }
        },
        ge!(x, 0.0) & ne!(x, 1.0)
    );

    impl_arb_op!(
        ln(x),
        if x.inf() > 0.0 && x.sup() < f64::INFINITY {
            arb_ln(x)
        } else {
            x.ln()
        },
        gt!(x, 0.0)
    );

    impl_arb_op!(
        log10(x),
        if x.inf() > 0.0 && x.sup() < f64::INFINITY {
            arb_log10(x)
        } else {
            x.log10()
        },
        gt!(x, 0.0)
    );

    impl_arb_op!(
        log2(x),
        if x.inf() > 0.0 && x.sup() < f64::INFINITY {
            arb_log2(x)
        } else {
            x.log2()
        },
        gt!(x, 0.0)
    );

    impl_arb_op!(shi(x), {
        let a = x.inf();
        let b = x.sup();
        if x.is_entire() {
            x
        } else if a == f64::NEG_INFINITY {
            // [-∞, Shi(b)]
            interval!(f64::NEG_INFINITY, arb_shi(i(b)).sup()).unwrap()
        } else if b == f64::INFINITY {
            // [Shi(a), +∞]
            interval!(arb_shi(i(a)).inf(), f64::INFINITY).unwrap()
        } else {
            arb_shi(x)
        }
    });

    impl_arb_op!(si(x), {
        let a = x.inf();
        let b = x.sup();
        if b <= 0.0 {
            arb_si(x).intersection(ci_envelope(-x) - Interval::FRAC_PI_2)
        } else if a >= 0.0 {
            arb_si(x).intersection(ci_envelope(x) + Interval::FRAC_PI_2)
        } else {
            arb_si(x)
        }
    });

    impl_arb_op!(sin(x), arb_sin(x));

    impl_arb_op!(
        sinc(x),
        if x.is_common_interval() {
            arb_sinc(x)
        } else {
            interval_set_ops::sinc(x)
        }
    );

    impl_arb_op!(
        sinh(x),
        if x.is_common_interval() {
            arb_sinh(x)
        } else {
            x.sinh()
        }
    );

    pub fn tan(&self, site: Option<Site>) -> Self {
        if self.iter().all(|x| {
            let a = x.x.inf();
            let b = x.x.sup();
            let q_nowrap = (x.x / Interval::FRAC_PI_2).floor();
            let qa = q_nowrap.inf();
            let qb = q_nowrap.sup();
            let n = if a == b { 0.0 } else { qb - qa };
            let q = qa.rem_euclid(2.0);
            q == 0.0 && n < 1.0 || q == 1.0 && n < 2.0
        }) {
            let mut rs = Self::new();
            for x in self {
                let dec = Decoration::Dac.min(x.d);
                let y = arb_tan(x.x);
                rs.insert(TupperInterval::new(DecInterval::set_dec(y, dec), x.g));
            }
            rs.normalize(false);
            rs
        } else {
            self.tan_impl(site)
        }
    }

    impl_arb_op!(
        tanh(x),
        if x.is_common_interval() {
            arb_tanh(x)
        } else {
            x.tanh()
        }
    );
}

macro_rules! arb_fn {
    ($f:ident($x:ident $(,$y:ident)*), $arb_f:ident($($args:expr),*), $range:expr) => {
        fn $f($x: Interval, $($y: Interval,)*) -> Interval {
            use crate::arb::Arb;
            let mut $x = Arb::from_interval($x);
            $(let mut $y = Arb::from_interval($y);)*
            unsafe {
                #[allow(unused_imports)]
                use std::ptr::null_mut as null;
                let $x = $x.as_mut_ptr();
                $(let $y = $y.as_mut_ptr();)*
                graphest_arb_sys::$arb_f($($args),*);
            }
            $x.to_interval().intersection($range)
        }
    };
}

macro_rules! acb_fn_reals {
    ($f:ident($x:ident $(,$y:ident)*), $acb_f:ident($($args:expr),*), $range:expr) => {
        fn $f($x: Interval, $($y: Interval,)*) -> Interval {
            use crate::arb::{Acb, Arb};
            let mut $x = Acb::from(Arb::from_interval($x));
            $(let mut $y = Acb::from(Arb::from_interval($y));)*
            unsafe {
                #[allow(unused_imports)]
                use std::ptr::null_mut as null;
                let $x = $x.as_mut_ptr();
                $(let $y = $y.as_mut_ptr();)*
                graphest_arb_sys::$acb_f($($args),*);
            }
            $x.real().to_interval().intersection($range)
        }
    };
}

arb_fn!(
    arb_acos(x),
    arb_acos(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(0.0, 3.1415926535897936) // [0, π]
);
arb_fn!(
    arb_acosh(x),
    arb_acosh(x, x, f64::MANTISSA_DIGITS.into()),
    ZERO_TO_INF
);
arb_fn!(
    arb_airy_ai(x),
    arb_hypgeom_airy(x, null(), null(), null(), x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.419015478032564, 0.5356566560156999)
);
arb_fn!(
    arb_airy_ai_prime(x),
    arb_hypgeom_airy(null(), x, null(), null(), x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_airy_bi(x),
    arb_hypgeom_airy(null(), null(), x, null(), x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.4549443836396574, f64::INFINITY)
);
arb_fn!(
    arb_airy_bi_prime(x),
    arb_hypgeom_airy(null(), null(), null(), x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_asin(x),
    arb_asin(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.5707963267948968, 1.5707963267948968) // [-π/2, π/2]
);
arb_fn!(
    arb_asinh(x),
    arb_asinh(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_atan(x),
    arb_atan(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.5707963267948968, 1.5707963267948968) // [-π/2, π/2]
);
arb_fn!(
    arb_atan2(y, x),
    arb_atan2(y, y, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-3.1415926535897936, 3.1415926535897936) // [-π, π]
);
arb_fn!(
    arb_atanh(x),
    arb_atanh(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_bessel_i(n, x),
    arb_hypgeom_bessel_i(n, n, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_bessel_j(n, x),
    arb_hypgeom_bessel_j(n, n, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_bessel_k(n, x),
    arb_hypgeom_bessel_k(n, n, x, f64::MANTISSA_DIGITS.into()),
    ZERO_TO_INF
);
arb_fn!(
    arb_bessel_y(n, x),
    arb_hypgeom_bessel_y(n, n, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_chi(x),
    arb_hypgeom_chi(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_ci(x),
    arb_hypgeom_ci(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(f64::NEG_INFINITY, 0.47200065143956865) // [-∞, Ci(π/2)]
);
arb_fn!(
    arb_cos(x),
    arb_cos(x, x, f64::MANTISSA_DIGITS.into()),
    M_ONE_TO_ONE
);
arb_fn!(
    arb_cosh(x),
    arb_cosh(x, x, f64::MANTISSA_DIGITS.into()),
    ONE_TO_INF
);
arb_fn!(
    arb_ei(x),
    arb_hypgeom_ei(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
acb_fn_reals!(
    arb_elliptic_e(x),
    acb_elliptic_e(x, x, f64::MANTISSA_DIGITS.into()),
    ONE_TO_INF
);
acb_fn_reals!(
    arb_elliptic_k(x),
    acb_elliptic_k(x, x, f64::MANTISSA_DIGITS.into()),
    ZERO_TO_INF
);
arb_fn!(
    arb_erf(x),
    // `+ 3` completes the graphing of "y = erf(1/x^21)".
    arb_hypgeom_erf(x, x, (f64::MANTISSA_DIGITS + 3).into()),
    M_ONE_TO_ONE
);
arb_fn!(
    arb_erfc(x),
    // `+ 3` completes the graphing of "y = erfc(1/x^21) + 1".
    arb_hypgeom_erfc(x, x, (f64::MANTISSA_DIGITS + 3).into()),
    const_interval!(0.0, 2.0)
);
arb_fn!(
    arb_erfi(x),
    arb_hypgeom_erfi(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_exp(x),
    arb_exp(x, x, f64::MANTISSA_DIGITS.into()),
    ZERO_TO_INF
);
arb_fn!(
    arb_exp10(x),
    arb_pow(
        x,
        Arb::from_f64(10.0).as_mut_ptr(), // TODO: `SyncLazy`
        x,
        f64::MANTISSA_DIGITS.into()
    ),
    ZERO_TO_INF
);
arb_fn!(
    arb_exp2(x),
    arb_pow(
        x,
        Arb::from_f64(2.0).as_mut_ptr(), // TODO: `SyncLazy`
        x,
        f64::MANTISSA_DIGITS.into()
    ),
    ZERO_TO_INF
);
arb_fn!(
    arb_fresnel_c(x),
    arb_hypgeom_fresnel(null(), x, x, 1, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.7798934003768229, 0.7798934003768229) // [C(-1), C(1)]
);
arb_fn!(
    arb_fresnel_s(x),
    arb_hypgeom_fresnel(x, null(), x, 1, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.7139722140219397, 0.7139722140219397) // [S(-√2), S(√2)]
);
arb_fn!(
    arb_gamma(x),
    arb_gamma(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_gamma_inc(a, x),
    arb_hypgeom_gamma_upper(a, a, x, 0, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_li(x),
    arb_hypgeom_li(x, x, 0, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_ln(x),
    arb_log(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_log10(x),
    arb_log_base_ui(x, x, 10, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_log2(x),
    arb_log_base_ui(x, x, 2, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_shi(x),
    arb_hypgeom_shi(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_si(x),
    arb_hypgeom_si(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.8519370519824663, 1.8519370519824663) // [Si(-π), Si(π)]
);
arb_fn!(
    arb_sin(x),
    arb_sin(x, x, f64::MANTISSA_DIGITS.into()),
    M_ONE_TO_ONE
);
arb_fn!(
    arb_sinc(x),
    arb_sinc(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.21723362821122166, 1.0)
);
arb_fn!(
    arb_sinh(x),
    arb_sinh(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_tan(x),
    arb_tan(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_tanh(x),
    arb_tanh(x, x, f64::MANTISSA_DIGITS.into()),
    M_ONE_TO_ONE
);

// Envelope functions

fn hypot(x: Interval, y: Interval) -> Interval {
    (x.sqr() + y.sqr()).sqrt()
}

/// Returns √[Ai(x)^2 + Bi(x)^2].
fn airy_envelope(x: Interval) -> Interval {
    let b = x.sup();
    let env = if b == f64::INFINITY {
        f64::INFINITY
    } else {
        let b = interval!(b, b).unwrap();
        hypot(arb_airy_ai(b), arb_airy_bi(b)).sup()
    };
    interval!(-env, env).unwrap()
}

/// Returns √[J_n(x)^2 + Y_n(x)^2].
fn bessel_envelope(n: Interval, x: Interval) -> Interval {
    let a = x.abs().inf();
    let a = interval!(a, a).unwrap();
    let env = hypot(arb_bessel_j(n, a), arb_bessel_y(n, a)).sup();
    interval!(-env, env).unwrap()
}

/// Returns √[Ci(x)^2 + si(x)^2], where si(x) = Si(x) - π/2.
///
/// Panics if `x.inf() < 0.0`.
fn ci_envelope(x: Interval) -> Interval {
    let a = x.inf();
    assert!(a >= 0.0);
    let a = interval!(a, a).unwrap();
    let env = hypot(arb_ci(a), arb_si(a) - Interval::FRAC_PI_2).sup();
    interval!(-env, env).unwrap()
}

/// Returns √[(C(x) - 1/2)^2 + (S(x) - 1/2)^2].
///
/// Panics if `x.inf() < 0.0`.
fn fresnel_envelope_centered(x: Interval) -> Interval {
    let a = x.inf();
    assert!(a >= 0.0);
    let a = interval!(a, a).unwrap();
    let env = hypot(arb_fresnel_c(a) - ONE_HALF, arb_fresnel_s(a) - ONE_HALF).sup();
    interval!(-env, env).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::*;

    #[test]
    fn arb_ops_sanity() {
        let xs = [
            TupperIntervalSet::from(const_dec_interval!(0.0, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(0.0, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(-1.0, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(-1.0, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(f64::NEG_INFINITY, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(f64::NEG_INFINITY, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(0.0, f64::INFINITY)),
            TupperIntervalSet::from(const_dec_interval!(1.0, f64::INFINITY)),
            TupperIntervalSet::from(DecInterval::ENTIRE),
        ];

        let fs = [
            TupperIntervalSet::airy_ai,
            TupperIntervalSet::airy_ai_prime,
            TupperIntervalSet::airy_bi,
            TupperIntervalSet::airy_bi_prime,
            TupperIntervalSet::chi,
            TupperIntervalSet::ci,
            TupperIntervalSet::ei,
            TupperIntervalSet::elliptic_e,
            TupperIntervalSet::elliptic_k,
            TupperIntervalSet::erfi,
            TupperIntervalSet::fresnel_c,
            TupperIntervalSet::fresnel_s,
            TupperIntervalSet::li,
            TupperIntervalSet::shi,
            TupperIntervalSet::si,
        ];
        for f in &fs {
            for x in &xs {
                f(x);
            }
        }

        let fs = [
            TupperIntervalSet::bessel_i,
            TupperIntervalSet::bessel_j,
            TupperIntervalSet::bessel_k,
            TupperIntervalSet::bessel_y,
            TupperIntervalSet::gamma_inc,
        ];
        let ns = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            .into_iter()
            .map(|n| TupperIntervalSet::from(dec_interval!(n, n).unwrap()))
            .collect::<Vec<_>>();
        for f in &fs {
            for n in &ns {
                for x in &xs {
                    f(n, x);
                }
            }
        }
    }
}
