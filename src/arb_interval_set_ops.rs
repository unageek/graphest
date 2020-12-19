use crate::{
    interval_set::{TupperInterval, TupperIntervalSet},
    interval_set_ops,
};
use inari::{const_interval, interval, DecInterval, Decoration, Interval};
use std::ops::{BitAnd, BitOr};

#[derive(Clone, Copy, Eq, Debug, PartialEq)]
struct BoolInterval(bool, bool);

impl BoolInterval {
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
    ($op:ident($x:ident), $result:expr, $def:expr) => {
        pub fn $op(&self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let $x = x.x;
                let def = $def;
                if def.possibly() {
                    let dec = if def.certainly() {
                        x.d
                    } else {
                        Decoration::Trv
                    };
                    rs.insert(TupperInterval::new(DecInterval::set_dec($result, dec), x.g));
                }
            }
            rs.normalize()
        }
    };

    ($op:ident($x:ident), $result:expr) => {
        impl_arb_op!($op($x), $result, BoolInterval::new(true, true));
    };

    ($op:ident($x:ident, $y:ident), $result:expr, $def:expr) => {
        pub fn $op(&self, rhs: &Self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.x;
                        let $y = y.x;
                        let def = $def;
                        if def.possibly() {
                            let dec = if def.certainly() {
                                x.d.min(y.d)
                            } else {
                                Decoration::Trv
                            };
                            rs.insert(TupperInterval::new(DecInterval::set_dec($result, dec), g));
                        }
                    }
                }
            }
            rs.normalize()
        }
    };
}

const M_ONE_TO_ONE: Interval = const_interval!(-1.0, 1.0);
const ONE_HALF: Interval = const_interval!(0.5, 0.5);
const ONE_TO_INF: Interval = const_interval!(1.0, f64::INFINITY);
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
            interval!(0.0, arb_airy_ai(interval!(a, a).unwrap()).sup()).unwrap()
        } else {
            arb_airy_ai(x).intersection(airy_envelope(x))
        }
    });
    impl_arb_op!(airy_ai_prime(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Ai'(a), 0]
            interval!(arb_airy_ai_prime(interval!(a, a).unwrap()).inf(), 0.0).unwrap()
        } else {
            arb_airy_ai_prime(x)
        }
    });
    impl_arb_op!(airy_bi(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Bi(a), +∞]
            interval!(arb_airy_bi(interval!(a, a).unwrap()).inf(), f64::INFINITY).unwrap()
        } else {
            arb_airy_bi(x).intersection(airy_envelope(x))
        }
    });
    impl_arb_op!(airy_bi_prime(x), {
        let a = x.inf();
        let b = x.sup();
        if a >= 0.0 && b == f64::INFINITY {
            // [Bi'(a), +∞]
            interval!(
                arb_airy_bi_prime(interval!(a, a).unwrap()).inf(),
                f64::INFINITY
            )
            .unwrap()
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
        bessel_j(n, x),
        arb_bessel_j(n, x).intersection(bessel_envelope(n, x)),
        {
            if !(n.is_singleton() && n.inf().fract() == 0.0) {
                panic!("Bessel functions of non-integer orders are not supprted");
            }
            BoolInterval::new(true, true)
        }
    );
    impl_arb_op!(
        bessel_y(n, x),
        {
            //              | -∞  if n ≥ 0 ∨ n ∈ (-1/2, 0] ∪ (-5/2, -3/2) ∪ (-9/2, -7/2) ∪ …,
            // lim Y_n(x) = | +∞  if n ∈ (-3/2, -1/2) ∪ (-7/2, -5/2) ∪ (-11/2, 9/2) ∪ …,
            // x→0⁺         | 0   otherwise (n = -1/2, -3/2, …).

            // Bisection at 1 is only valid for integer orders.
            // The inflection point can be arbitrarily close to 0 for general orders.
            let y0 = {
                let x = x.intersection(ZERO_TO_ONE);
                if x.is_empty() {
                    x
                } else {
                    let a = x.inf();
                    let b = x.sup();
                    let a = interval!(a, a).unwrap();
                    let b = interval!(b, b).unwrap();
                    if n.inf() >= 0.0 || n.inf() % 2.0 == 0.0 {
                        interval!(arb_bessel_y(n, a).inf(), arb_bessel_y(n, b).sup()).unwrap()
                    } else {
                        interval!(arb_bessel_y(n, b).inf(), arb_bessel_y(n, a).sup()).unwrap()
                    }
                }
            };
            let y1 = {
                let x = x.intersection(ONE_TO_INF);
                if x.is_empty() {
                    x
                } else {
                    arb_bessel_y(n, x).intersection(bessel_envelope(n, x))
                }
            };
            y0.convex_hull(y1)
        },
        {
            if !(n.is_singleton() && n.inf().fract() == 0.0) {
                panic!("Bessel functions of non-integer orders are not supprted");
            }
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
                interval!(f64::NEG_INFINITY, arb_chi(interval!(b, b).unwrap()).sup()).unwrap()
            } else if b == f64::INFINITY {
                // [Chi(a), +∞]
                interval!(arb_chi(interval!(a, a).unwrap()).inf(), f64::INFINITY).unwrap()
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
                let sup = arb_ci(interval!(b, b).unwrap()).sup();
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
                let inf = arb_ei(interval!(b, b).unwrap()).inf();
                let sup = if a == f64::NEG_INFINITY {
                    0.0
                } else {
                    arb_ei(interval!(a, a).unwrap()).sup()
                };
                interval!(inf, sup).unwrap()
            } else if a >= 0.0 {
                // [Ei(a), Ei(b)]
                let inf = arb_ei(interval!(a, a).unwrap()).inf();
                let sup = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_ei(interval!(b, b).unwrap()).sup()
                };
                interval!(inf, sup).unwrap()
            } else {
                // [-∞, max(Ei(a), Ei(b))]
                let sup0 = if a == f64::NEG_INFINITY {
                    0.0
                } else {
                    arb_ei(interval!(a, a).unwrap()).sup()
                };
                let sup1 = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_ei(interval!(b, b).unwrap()).sup()
                };
                interval!(f64::NEG_INFINITY, sup0.max(sup1)).unwrap()
            }
        },
        ne!(x, 0.0)
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
            interval!(f64::NEG_INFINITY, arb_erfi(interval!(b, b).unwrap()).sup()).unwrap()
        } else if b == f64::INFINITY {
            // [erfi(a), +∞]
            interval!(arb_erfi(interval!(a, a).unwrap()).inf(), f64::INFINITY).unwrap()
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
    impl_arb_op!(
        li(x),
        {
            let x = x.intersection(ZERO_TO_INF);
            let a = x.inf();
            let b = x.sup();
            if b <= 1.0 {
                // [li(b), li(a)]
                let inf = arb_li(interval!(b, b).unwrap()).inf();
                let sup = arb_li(interval!(a, a).unwrap()).sup();
                interval!(inf, sup).unwrap()
            } else if a >= 1.0 {
                // [li(a), li(b)]
                let inf = arb_li(interval!(a, a).unwrap()).inf();
                let sup = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_li(interval!(b, b).unwrap()).sup()
                };
                interval!(inf, sup).unwrap()
            } else {
                // [-∞, max(li(a), li(b))]
                let sup0 = arb_li(interval!(a, a).unwrap()).sup();
                let sup1 = if b == f64::INFINITY {
                    f64::INFINITY
                } else {
                    arb_li(interval!(b, b).unwrap()).sup()
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
            interval!(f64::NEG_INFINITY, arb_shi(interval!(b, b).unwrap()).sup()).unwrap()
        } else if b == f64::INFINITY {
            // [Shi(a), +∞]
            interval!(arb_shi(interval!(a, a).unwrap()).inf(), f64::INFINITY).unwrap()
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
                let $x = $x.as_raw_mut();
                $(let $y = $y.as_raw_mut();)*
                crate::arb_sys::$arb_f($($args),*);
            }
            $x.to_interval().intersection($range)
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
    arb_atanh(x),
    arb_atanh(x, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
);
arb_fn!(
    arb_bessel_j(n, x),
    arb_hypgeom_bessel_j(n, n, x, f64::MANTISSA_DIGITS.into()),
    Interval::ENTIRE
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
        Arb::from_f64(10.0).as_raw_mut(), // TODO: lazy_static
        x,
        f64::MANTISSA_DIGITS.into()
    ),
    ZERO_TO_INF
);
arb_fn!(
    arb_exp2(x),
    arb_pow(
        x,
        Arb::from_f64(2.0).as_raw_mut(), // TODO: lazy_static
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
    arb_tanh(x),
    arb_tanh(x, x, f64::MANTISSA_DIGITS.into()),
    M_ONE_TO_ONE
);

// Envelope functions
fn hypot(x: Interval, y: Interval) -> Interval {
    (x.sqr() + y.sqr()).sqrt()
}
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
fn bessel_envelope(n: Interval, x: Interval) -> Interval {
    let a = x.abs().inf();
    let a = interval!(a, a).unwrap();
    let env = hypot(arb_bessel_j(n, a), arb_bessel_y(n, a)).sup();
    interval!(-env, env).unwrap()
}
fn ci_envelope(x: Interval) -> Interval {
    let a = x.inf();
    assert!(a >= 0.0);
    let a = interval!(a, a).unwrap();
    let env = hypot(arb_ci(a), arb_si(a) - Interval::FRAC_PI_2).sup();
    interval!(-env, env).unwrap()
}
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
        // Just check that arb ops don't panic due to invalid construction of an interval.
        let xs = [
            TupperIntervalSet::from(const_dec_interval!(f64::NEG_INFINITY, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(0.0, f64::INFINITY)),
            TupperIntervalSet::from(DecInterval::ENTIRE),
        ];

        let fs = [
            TupperIntervalSet::acos,
            TupperIntervalSet::acosh,
            TupperIntervalSet::airy_ai,
            TupperIntervalSet::airy_ai_prime,
            TupperIntervalSet::airy_bi,
            TupperIntervalSet::airy_bi_prime,
            TupperIntervalSet::asin,
            TupperIntervalSet::asinh,
            TupperIntervalSet::atan,
            TupperIntervalSet::atanh,
            TupperIntervalSet::chi,
            TupperIntervalSet::ci,
            TupperIntervalSet::cos,
            TupperIntervalSet::cosh,
            TupperIntervalSet::ei,
            TupperIntervalSet::erf,
            TupperIntervalSet::erfc,
            TupperIntervalSet::erfi,
            TupperIntervalSet::exp,
            TupperIntervalSet::exp10,
            TupperIntervalSet::exp2,
            TupperIntervalSet::fresnel_c,
            TupperIntervalSet::fresnel_s,
            TupperIntervalSet::li,
            TupperIntervalSet::ln,
            TupperIntervalSet::log10,
            TupperIntervalSet::log2,
            TupperIntervalSet::shi,
            TupperIntervalSet::si,
            TupperIntervalSet::sin,
            TupperIntervalSet::sinc,
            TupperIntervalSet::sinh,
            TupperIntervalSet::tanh,
        ];
        for f in fs.iter() {
            for x in xs.iter() {
                f(x);
            }
        }

        let fs = [TupperIntervalSet::bessel_j, TupperIntervalSet::bessel_y];
        let ns = (-2..=2)
            .map(|n| TupperIntervalSet::from(dec_interval!(n as f64, n as f64).unwrap()))
            .collect::<Vec<_>>();
        for f in fs.iter() {
            for (n, x) in ns.iter().zip(xs.iter()) {
                f(n, x);
            }
        }
    }
}
