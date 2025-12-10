use crate::{interval_set::TupperIntervalSet, rational_ops};
use inari::{const_interval, DecInterval, Decoration, Interval};
use rug::Rational;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RealUnit {
    One,
    Pi,
}

/// Stores the value of an AST node of kind [`ExprKind::Constant`].
/// Values that are elements of ℚ ∪ π ℚ can be represented exactly.
///
/// [`ExprKind::Constant`]: crate::ast::ExprKind::Constant
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Real {
    x: TupperIntervalSet,
    q: Option<(Rational, RealUnit)>,
}

fn interval_set(x: Interval) -> TupperIntervalSet {
    // Always decorate with `Decoration::Dac` to avoid that, for example,
    // `Real::zero().floor().interval().decoration()` results in `Decoration::Com`,
    // which should actually be `Decoration::Dac`.
    TupperIntervalSet::from(DecInterval::set_dec(x, Decoration::Dac))
}

impl Real {
    /// Returns a [`Real`] representing 0, exactly.
    pub fn zero() -> Self {
        Self {
            x: interval_set(const_interval!(0.0, 0.0)),
            q: Some((Rational::new(), RealUnit::One)),
        }
    }

    /// Returns a [`Real`] representing 1, exactly.
    pub fn one() -> Self {
        Self {
            x: interval_set(const_interval!(1.0, 1.0)),
            q: Some((Rational::from(1), RealUnit::One)),
        }
    }

    /// Returns a [`Real`] representing π, exactly.
    pub fn pi() -> Self {
        Self {
            x: interval_set(Interval::PI),
            q: Some((Rational::from(1), RealUnit::Pi)),
        }
    }

    /// Returns a [`Real`] representing 2π, exactly.
    pub fn tau() -> Self {
        Self {
            x: interval_set(Interval::TAU),
            q: Some((Rational::from(2), RealUnit::Pi)),
        }
    }

    /// Returns an enclosure of the value.
    pub fn interval(&self) -> &TupperIntervalSet {
        &self.x
    }

    /// Returns the value as [`Rational`] if the representation is exact.
    pub fn rational(&self) -> Option<&Rational> {
        match &self.q {
            Some((q, RealUnit::One)) => Some(q),
            Some((q, RealUnit::Pi)) if q.is_zero() => Some(q),
            _ => None,
        }
    }

    /// Returns the value as [`Rational`] multiple of π if the representation is exact.
    pub fn rational_pi(&self) -> Option<&Rational> {
        match &self.q {
            Some((q, RealUnit::Pi)) => Some(q),
            Some((q, RealUnit::One)) if q.is_zero() => Some(q),
            _ => None,
        }
    }

    /// Returns the value as [`Rational`] and its unit if the representation is exact.
    pub fn rational_unit(&self) -> Option<(&Rational, RealUnit)> {
        self.q.as_ref().map(|(q, unit)| (q, *unit))
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
        Self {
            x: interval_set(rational_ops::to_interval(&q)),
            q: Some((q, RealUnit::One)),
        }
    }
}

impl From<(Rational, RealUnit)> for Real {
    fn from((q, unit): (Rational, RealUnit)) -> Self {
        if q.is_zero() {
            return Self::zero();
        }

        Self {
            x: match unit {
                RealUnit::One => interval_set(rational_ops::to_interval(&q)),
                RealUnit::Pi => interval_set(rational_ops::to_interval(&q) * Interval::PI),
            },
            q: Some((q, unit)),
        }
    }
}

impl From<TupperIntervalSet> for Real {
    fn from(x: TupperIntervalSet) -> Self {
        let q = x
            .to_f64()
            .and_then(Rational::from_f64)
            .map(|q| (q, RealUnit::One));
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

    ($op:ident($x:ident), $y:expr, $y_q:expr, $y_q_pi:expr) => {
        pub fn $op(self) -> Self {
            let y_q = match self.q {
                Some(($x, RealUnit::One)) => $y_q.map(|q| (q, RealUnit::One)),
                Some(($x, RealUnit::Pi)) => $y_q_pi.map(|q| (q, RealUnit::Pi)),
                _ => None,
            };
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

    ($op:ident($x:ident, $y:ident), $z:expr, $z_q:expr, $z_q_pi:expr) => {
        pub fn $op(self, rhs: Self) -> Self {
            let z_q = match (self.q, rhs.q) {
                (Some(($x, RealUnit::One)), Some(($y, RealUnit::One))) => {
                    $z_q.map(|q| (q, RealUnit::One))
                }
                (Some(($x, RealUnit::Pi)), Some(($y, RealUnit::Pi))) => {
                    $z_q_pi.map(|q| (q, RealUnit::Pi))
                }
                _ => None,
            };
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

#[allow(unused_variables)]
impl Real {
    impl_op!(abs(x), x.abs(), Some(x.abs()), Some(x.abs()));
    impl_op!(acos(x));
    impl_op!(acosh(x));

    pub fn add(self, rhs: Self) -> Self {
        let z_q = match (self.q, rhs.q) {
            (Some((x, x_unit)), Some((y, y_unit))) if x_unit == y_unit => Some((x + y, x_unit)),
            (Some((x, _)), y_q) | (y_q, Some((x, _))) if x.is_zero() => y_q,
            _ => None,
        };
        if let Some(z_q) = z_q {
            z_q.into()
        } else {
            let x = self.x;
            let y = rhs.x;
            (&x + &y).into()
        }
    }

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
    impl_op!(ceil(x), x.ceil(None), Some(x.ceil()), None);
    impl_op!(chi(x));
    impl_op!(ci(x));

    pub fn cos(self) -> Self {
        let y_q = match self.q {
            Some((x, RealUnit::Pi)) => rational_ops::cos_pi(x).map(|q| (q, RealUnit::One)),
            _ => None,
        };
        if let Some(y_q) = y_q {
            y_q.into()
        } else {
            let x = self.x;
            x.cos().into()
        }
    }

    impl_op!(cosh(x));
    impl_op!(digamma(x), x.digamma(None));

    pub fn div(self, rhs: Self) -> Self {
        let z_q = match (self.q, rhs.q) {
            (Some((x, x_unit)), Some((y, y_unit))) if x_unit == y_unit => {
                rational_ops::div(x, y).map(|q| (q, RealUnit::One))
            }
            (Some((x, RealUnit::Pi)), Some((y, RealUnit::One))) => {
                rational_ops::div(x, y).map(|q| (q, RealUnit::Pi))
            }
            _ => None,
        };
        if let Some(z_q) = z_q {
            z_q.into()
        } else {
            let x = self.x;
            let y = rhs.x;
            x.div(&y, None).into()
        }
    }

    impl_op!(ei(x));
    impl_op!(elliptic_e(x));
    impl_op!(elliptic_k(x));
    impl_op!(erf(x));
    impl_op!(erfc(x));
    impl_op!(erfi(x));
    impl_op!(exp(x));
    impl_op!(floor(x), x.floor(None), Some(x.floor()), None);
    impl_op!(fresnel_c(x));
    impl_op!(fresnel_s(x));
    impl_op!(gamma(x), x.gamma(None));
    impl_op!(gamma_inc(a, x));
    impl_op!(gcd(x, y), x.gcd(&y, None), rational_ops::gcd(x, y), None);
    impl_op!(if_then_else(cond, t, f));
    impl_op!(im_sinc(re_x, im_x));
    impl_op!(im_undef_at_0(re_x, im_x));
    impl_op!(im_zeta(re_x, im_x));
    impl_op!(inverse_erf(x));
    impl_op!(inverse_erfc(x));
    impl_op!(lambert_w(k, x));
    impl_op!(lcm(x, y), x.lcm(&y, None), rational_ops::lcm(x, y), None);
    impl_op!(li(x));
    impl_op!(ln(x));
    impl_op!(ln_gamma(x));
    impl_op!(log(x, b), x.log(&b, None));
    impl_op!(
        max(x, y),
        x.max(&y),
        rational_ops::max(x, y),
        rational_ops::max(x, y)
    );
    impl_op!(
        min(x, y),
        x.min(&y),
        rational_ops::min(x, y),
        rational_ops::min(x, y)
    );
    impl_op!(
        modulo(x, y),
        x.modulo(&y, None),
        rational_ops::modulo(x, y),
        rational_ops::modulo(x, y)
    );

    pub fn mul(self, rhs: Self) -> Self {
        let z_q = match (self.q, rhs.q) {
            (Some((x, RealUnit::One)), Some((y, RealUnit::One))) => Some((x * y, RealUnit::One)),
            (Some((x, RealUnit::One)), Some((y, RealUnit::Pi)))
            | (Some((x, RealUnit::Pi)), Some((y, RealUnit::One))) => Some((x * y, RealUnit::Pi)),
            _ => None,
        };
        if let Some(z_q) = z_q {
            z_q.into()
        } else {
            let x = self.x;
            let y = rhs.x;
            (&x * &y).into()
        }
    }

    impl_op!(neg(x), -&x, Some(-x), Some(-x));
    impl_op!(pow(x, y), x.pow(&y, None), rational_ops::pow(x, y), None);
    impl_op!(
        pow_rational(x, y),
        x.pow_rational(&y, None),
        rational_ops::pow_rational(x, y),
        None
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

    pub fn sin(self) -> Self {
        let y_q = match self.q {
            Some((x, RealUnit::Pi)) => rational_ops::sin_pi(x).map(|q| (q, RealUnit::One)),
            _ => None,
        };
        if let Some(y_q) = y_q {
            y_q.into()
        } else {
            let x = self.x;
            x.sin().into()
        }
    }

    impl_op!(sinc(x));
    impl_op!(sinh(x));

    pub fn sub(self, rhs: Self) -> Self {
        let z_q = match (self.q, rhs.q) {
            (Some((x, x_unit)), Some((y, y_unit))) if x_unit == y_unit => Some((x - y, x_unit)),
            (Some((x, _)), Some((y, unit))) if x.is_zero() => Some((-y, unit)),
            (x_q, Some((y, _))) if y.is_zero() => x_q,
            _ => None,
        };
        if let Some(z_q) = z_q {
            z_q.into()
        } else {
            let x = self.x;
            let y = rhs.x;
            (&x - &y).into()
        }
    }

    pub fn tan(self) -> Self {
        let y_q = match self.q {
            Some((x, RealUnit::Pi)) => rational_ops::tan_pi(x).map(|q| (q, RealUnit::One)),
            _ => None,
        };
        if let Some(y_q) = y_q {
            y_q.into()
        } else {
            let x = self.x;
            x.tan(None).into()
        }
    }

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
    fn decoration() {
        let x = Real::zero();
        let y = x.floor();
        // Without the treatment explained in `interval_set`,
        // the decoration would be `Decoration::Com`, which is wrong.
        assert_eq!(y.interval().decoration(), Decoration::Dac);
    }

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
