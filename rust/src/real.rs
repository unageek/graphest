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
    pub fn rational(&self) -> Option<&Rational> {
        self.q.as_ref()
    }

    /// Returns the value as [`f64`] if the representation is exact.
    pub fn to_f64(&self) -> Option<f64> {
        self.x.to_f64()
    }
}

impl From<DecInterval> for Real {
    fn from(x: DecInterval) -> Self {
        let x = TupperIntervalSet::from(x);
        Self::from(x)
    }
}

impl From<Rational> for Real {
    fn from(q: Rational) -> Self {
        // Always use `Decoration::Dac` to avoid the value of `floor([0]_com)`
        // becoming `[0]_com` instead of `[0]_dac`, for example.
        let x = TupperIntervalSet::from(DecInterval::set_dec(
            rational_ops::to_interval(&q),
            Decoration::Dac,
        ));
        Self { x, q: Some(q) }
    }
}

impl From<TupperIntervalSet> for Real {
    fn from(x: TupperIntervalSet) -> Self {
        let q = x.to_f64().and_then(Rational::from_f64);
        Self { x, q }
    }
}

macro_rules! impl_op {
    ($op:ident($x:ident)) => {
        impl_op!($op($x), $x.$op());
    };

    ($op:ident($x:ident), $y:expr) => {
        pub fn $op(self) -> Self {
            let $x = self.x;
            $y.into()
        }
    };

    ($op:ident($x:ident), $y:expr, $y_q:expr) => {
        pub fn $op(self) -> Self {
            let y_q = self.q.and_then(|$x| $y_q);
            if let Some(y_q) = y_q {
                y_q.into()
            } else {
                let $x = self.x;
                $y.into()
            }
        }
    };

    ($op:ident($x:ident, $y:ident)) => {
        impl_op!($op($x, $y), $x.$op(&$y));
    };

    ($op:ident($x:ident, $y:ident), $z:expr) => {
        pub fn $op(self, rhs: Self) -> Self {
            let $x = self.x;
            let $y = rhs.x;
            $z.into()
        }
    };

    ($op:ident($x:ident, $y:ident), $z:expr, $z_q:expr) => {
        pub fn $op(self, rhs: Self) -> Self {
            let z_q = self.q.zip(rhs.q).and_then(|($x, $y)| $z_q);
            if let Some(z_q) = z_q {
                z_q.into()
            } else {
                let $x = self.x;
                let $y = rhs.x;
                $z.into()
            }
        }
    };

    ($op:ident($x:ident, $y:ident, $z:ident)) => {
        impl_op!($op($x, $y, $z), $x.$op(&$y, &$z));
    };

    ($op:ident($x:ident, $y:ident, $z:ident), $result:expr) => {
        pub fn $op(self, $y: Self, $z: Self) -> Self {
            let $x = self.x;
            let $y = $y.x;
            let $z = $z.x;
            $result.into()
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
    impl_op!(boole_eq_zero(x), x.boole_eq_zero(None));
    impl_op!(boole_le_zero(x), x.boole_le_zero(None));
    impl_op!(boole_lt_zero(x), x.boole_lt_zero(None));
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
    impl_op!(if_then_else(cond, t, f));
    impl_op!(im_sinc(re_x, im_x));
    impl_op!(im_undef_at_0(re_x, im_x));
    impl_op!(im_zeta(re_x, im_x));
    impl_op!(inverse_erf(x));
    impl_op!(inverse_erfc(x));
    impl_op!(lambert_w(k, x));
    impl_op!(lcm(x, y), x.lcm(&y, None), rational_ops::lcm(x, y));
    impl_op!(li(x));
    impl_op!(ln(x));
    impl_op!(ln_gamma(x));
    impl_op!(log(x, b), x.log(&b, None));
    impl_op!(max(x, y), x.max(&y), rational_ops::max(x, y));
    impl_op!(min(x, y), x.min(&y), rational_ops::min(x, y));
    impl_op!(modulo(x, y), x.modulo(&y, None), rational_ops::modulo(x, y));
    impl_op!(mul(x, y), &x * &y, Some(x * y));
    impl_op!(neg(x), -&x, Some(-x));
    impl_op!(pow(x, y), x.pow(&y, None), rational_ops::pow(x, y));
    impl_op!(
        pow_rational(x, y),
        x.pow_rational(&y, None),
        rational_ops::pow_rational(x, y)
    );

    pub fn ranked_max(xs: Vec<Real>, n: Real) -> Self {
        TupperIntervalSet::ranked_max(xs.iter().map(|x| &x.x).collect(), &n.x, None).into()
    }

    pub fn ranked_min(xs: Vec<Real>, n: Real) -> Self {
        TupperIntervalSet::ranked_min(xs.iter().map(|x| &x.x).collect(), &n.x, None).into()
    }

    impl_op!(re_sign_nonnegative(x, y), x.re_sign_nonnegative(&y, None));
    impl_op!(re_sinc(re_x, im_x));
    impl_op!(re_undef_at_0(re_x, im_x));
    impl_op!(re_zeta(re_x, im_x));
    impl_op!(shi(x));
    impl_op!(si(x));
    impl_op!(sin(x));
    impl_op!(sinc(x));
    impl_op!(sinh(x));
    impl_op!(sub(x, y), &x - &y, Some(x - y));
    impl_op!(tan(x), x.tan(None));
    impl_op!(tanh(x));
    impl_op!(undef_at_0(x));
    impl_op!(zeta(x));
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
    use super::*;
    use inari::{const_dec_interval, const_interval};

    #[test]
    fn from_dec_interval() {
        let x = Real::from(const_dec_interval!(1.5, 1.5));
        assert_eq!(x.rational(), Some(&(3, 2).into()));
        assert_eq!(x.to_f64(), Some(1.5));

        let x = Real::from(DecInterval::PI);
        assert_eq!(x.rational(), None);
        assert_eq!(x.to_f64(), None);
    }

    #[test]
    fn from_rational() {
        let x = Real::from(Rational::from((3, 2)));
        assert_eq!(
            *x.interval(),
            TupperIntervalSet::from(DecInterval::set_dec(
                const_interval!(1.5, 1.5),
                Decoration::Dac
            ))
        );
        assert_eq!(x.to_f64(), Some(1.5));
    }
}
