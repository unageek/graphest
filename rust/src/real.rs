use crate::{interval_set::TupperIntervalSet, rational_ops};
use inari::{DecInterval, Decoration};
use rug::Rational;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Stores the value of an AST node of kind [`ExprKind::Constant`].
///
/// [`ExprKind::Constant`]: crate::ast::ExprKind::Constant
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Real {
    x: TupperIntervalSet,
    q: Option<Rational>,
}

impl Real {
    /// Returns an enclosure of the value.
    pub fn interval(&self) -> &TupperIntervalSet {
        &self.x
    }

    /// Returns the value as [`Rational`] if the representation is exact.
    pub fn rational(&self) -> &Option<Rational> {
        &self.q
    }

    /// Returns the value as [`f64`] if the representation is exact.
    pub fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl Real {
    pub fn new(x: TupperIntervalSet, q: Option<Rational>) -> Self {
        let q = q.or_else(|| x.to_f64().and_then(Rational::from_f64));
        Self { x, q }
    }
}

macro_rules! impl_op {
    ($op:ident($x:ident)) => {
        impl_op!($op($x), $x.$op());
    };

    ($op:ident($x:ident), $y:expr) => {
        pub fn $op(self) -> Self {
            let y = {
                let $x = self.x;
                $y
            };
            Self::new(y, None)
        }
    };

    ($op:ident($x:ident), $y:expr, $y_q:expr) => {
        pub fn $op(self) -> Self {
            let y_q = self.q.and_then(|$x| $y_q);
            let y = if let Some(y_q) = &y_q {
                // Always use `Decoration::Dac` to avoid `floor([0]_com)`
                // being evaluated to `[0]_com` instead of `[0]_dac`, for example.
                TupperIntervalSet::from(DecInterval::set_dec(
                    rational_ops::to_interval(y_q),
                    Decoration::Dac,
                ))
            } else {
                let $x = self.x;
                $y
            };
            Self::new(y, y_q)
        }
    };

    ($op:ident($x:ident, $y:ident)) => {
        impl_op!($op($x, $y), $x.$op(&$y));
    };

    ($op:ident($x:ident, $y:ident), $z:expr) => {
        pub fn $op(self, rhs: Self) -> Self {
            let z = {
                let $x = self.x;
                let $y = rhs.x;
                $z
            };
            Self::new(z, None)
        }
    };

    ($op:ident($x:ident, $y:ident), $z:expr, $z_q:expr) => {
        pub fn $op(self, rhs: Self) -> Self {
            let z_q = self.q.zip(rhs.q).and_then(|($x, $y)| $z_q);
            let z = if let Some(z_q) = &z_q {
                TupperIntervalSet::from(DecInterval::set_dec(
                    rational_ops::to_interval(z_q),
                    Decoration::Dac,
                ))
            } else {
                let $x = self.x;
                let $y = rhs.x;
                $z
            };
            Self::new(z, z_q)
        }
    };
}

impl Real {
    impl_op!(abs(x), x.abs(), Some(x.abs()));
    impl_op!(acos(x));
    impl_op!(acosh(x));
    impl_op!(add(x, y), &x + &y, Some(x + y));
    impl_op!(airy_ai(x));
    impl_op!(airy_ai_prime(x));
    impl_op!(airy_bi(x));
    impl_op!(airy_bi_prime(x));
    impl_op!(asin(x));
    impl_op!(asinh(x));
    impl_op!(atan(x));
    impl_op!(atanh(x));
    impl_op!(atan2(y, x), y.atan2(&x, None));
    impl_op!(bessel_i(n, x));
    impl_op!(bessel_j(n, x));
    impl_op!(bessel_k(n, x));
    impl_op!(bessel_y(n, x));
    impl_op!(ceil(x), x.ceil(None), Some(x.ceil()));
    impl_op!(chi(x));
    impl_op!(ci(x));
    impl_op!(cos(x));
    impl_op!(cosh(x));
    impl_op!(digamma(x), x.digamma(None));
    impl_op!(div(x, y), x.div(&y, None), rational_ops::div(x, y));
    impl_op!(ei(x));
    impl_op!(elliptic_e(x));
    impl_op!(elliptic_k(x));
    impl_op!(erf(x));
    impl_op!(erfc(x));
    impl_op!(erfi(x));
    impl_op!(exp(x));
    impl_op!(floor(x), x.floor(None), Some(x.floor()));
    impl_op!(fresnel_c(x));
    impl_op!(fresnel_s(x));
    impl_op!(gamma(x), x.gamma(None));
    impl_op!(gamma_inc(a, x));
    impl_op!(gcd(x, y), x.gcd(&y, None), rational_ops::gcd(x, y));
    impl_op!(lcm(x, y), x.lcm(&y, None), rational_ops::lcm(x, y));
    impl_op!(li(x));
    impl_op!(ln(x));
    impl_op!(log(x, b), x.log(&b, None));
    impl_op!(log10(x));
    impl_op!(max(x, y), x.max(&y), rational_ops::max(x, y));
    impl_op!(min(x, y), x.min(&y), rational_ops::min(x, y));
    impl_op!(mul(x, y), &x * &y, Some(x * y));
    impl_op!(neg(x), -&x, Some(-x));
    impl_op!(one(x));
    impl_op!(pow(x, y), x.pow(&y, None), rational_ops::pow(x, y));

    pub fn ranked_max(xs: Vec<Real>, n: Real) -> Self {
        let y = TupperIntervalSet::ranked_max(xs.iter().map(|x| &x.x).collect(), &n.x, None);
        Self::new(y, None)
    }

    pub fn ranked_min(xs: Vec<Real>, n: Real) -> Self {
        let y = TupperIntervalSet::ranked_min(xs.iter().map(|x| &x.x).collect(), &n.x, None);
        Self::new(y, None)
    }

    impl_op!(
        rem_euclid(x, y),
        x.rem_euclid(&y, None),
        rational_ops::rem_euclid(x, y)
    );

    pub fn rootn(self, n: u32) -> Self {
        let y = self.x.rootn(n);
        Self::new(y, None)
    }

    impl_op!(shi(x));
    impl_op!(si(x));
    impl_op!(sin(x));
    impl_op!(sinc(x));
    impl_op!(sinh(x));
    impl_op!(sub(x, y), &x - &y, Some(x - y));
    impl_op!(sqr(x), x.sqr(), Some(x.square()));
    impl_op!(sqrt(x));
    impl_op!(tan(x), x.tan(None));
    impl_op!(tanh(x));
    impl_op!(undef_at_0(x));
}

impl Neg for Real {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

impl Add for Real {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub for Real {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul for Real {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl Div for Real {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

#[cfg(test)]
mod tests {
    use inari::const_dec_interval;

    use super::*;

    #[test]
    fn real() {
        let x = Real::new(const_dec_interval!(1.5, 1.5).into(), None);
        assert_eq!(*x.rational(), Some((3, 2).into()));

        let x = Real::new(DecInterval::PI.into(), None);
        assert_eq!(*x.rational(), None);
    }
}
