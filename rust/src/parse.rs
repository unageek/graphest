use crate::{
    ast::{Expr, NaryOp},
    context::{Context, InputWithContext},
    real::Real,
};
use inari::dec_interval;
use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_while},
    character::complete::{char, digit0, digit1, one_of, satisfy, space0},
    combinator::{
        all_consuming, consumed, cut, map, map_opt, not, opt, peek, recognize, value, verify,
    },
    error::{ErrorKind as NomErrorKind, ParseError},
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    Err, Finish, IResult, InputLength, Parser,
};
use rug::{Integer, Rational};
use std::ops::Range;

#[derive(Clone, Debug)]
enum ErrorKind<'a> {
    ExpectedChar(char),
    ExpectedEof,
    ExpectedExpr,
    UnknownIdentifier(&'a str),
    /// Errors reported by nom's combinators that should not be exposed.
    OtherNomError,
}

#[derive(Clone, Debug)]
struct Error<'a, I> {
    input: I,
    kind: ErrorKind<'a>,
}

impl<'a, I> Error<'a, I> {
    fn expected_expr(input: I) -> Self {
        Self {
            input,
            kind: ErrorKind::ExpectedExpr,
        }
    }

    fn unknown_identifier(input: I, name: &'a str) -> Self {
        Self {
            input,
            kind: ErrorKind::UnknownIdentifier(name),
        }
    }
}

impl<'a, I> ParseError<I> for Error<'a, I> {
    fn append(_: I, _: NomErrorKind, other: Self) -> Self {
        // Only keep the first error.
        other
    }

    fn from_char(input: I, c: char) -> Self {
        Self {
            input,
            kind: ErrorKind::ExpectedChar(c),
        }
    }

    fn from_error_kind(input: I, kind: NomErrorKind) -> Self {
        Self {
            input,
            kind: match kind {
                NomErrorKind::Eof => ErrorKind::ExpectedEof,
                _ => ErrorKind::OtherNomError,
            },
        }
    }
}

type ParseResult<'a, O> = IResult<InputWithContext<'a>, O, Error<'a, InputWithContext<'a>>>;

// Based on `inari::parse::parse_dec_float`.
fn parse_decimal(mant: &str) -> Option<Rational> {
    fn pow(base: u32, exp: i32) -> Rational {
        let i = Integer::from(Integer::u_pow_u(base, exp.unsigned_abs()));
        let mut r = Rational::from(i);
        if exp < 0 {
            r.recip_mut();
        }
        r
    }

    let mut parts = mant.split('.');
    let int_part = parts.next().unwrap();
    let frac_part = match parts.next() {
        Some(s) => s,
        _ => "",
    };

    // 123.456 -> 123456e-3 (ulp == 1e-3)
    let log_ulp = -(frac_part.len() as i32);
    let ulp = pow(10, log_ulp);

    let i_str = [int_part, frac_part].concat();
    let i = Integer::parse_radix(i_str, 10).unwrap();
    Some(Rational::from(i) * ulp)
}

fn decimal_literal(i: InputWithContext) -> ParseResult<&str> {
    map(
        alt((
            // "12", "12." or "12.3"
            recognize(pair(digit1, opt(pair(char('.'), digit0)))),
            // ".3"
            recognize(pair(char('.'), digit1)),
        )),
        |i: InputWithContext| i.source,
    )(i)
}

fn decimal_constant(i: InputWithContext) -> ParseResult<Expr> {
    map(decimal_literal, |s| {
        let x = if let Some(x_q) = parse_decimal(s) {
            Real::from(x_q)
        } else {
            let interval_literal = ["[", s, "]"].concat();
            Real::from(dec_interval!(&interval_literal).unwrap())
        };
        Expr::constant(x)
    })(i)
}

fn identifier_head(i: InputWithContext) -> ParseResult<char> {
    satisfy(|c| c.is_alphabetic())(i)
}

fn identifier_tail(i: InputWithContext) -> ParseResult<&str> {
    map(
        recognize(many0_count(satisfy(|c| c.is_alphanumeric() || c == '\''))),
        |i: InputWithContext| i.source,
    )(i)
}

fn identifier(i: InputWithContext) -> ParseResult<&str> {
    map(recognize(pair(identifier_head, identifier_tail)), |i| {
        i.source
    })(i)
}

fn name_in_context(i: InputWithContext) -> ParseResult<(&Context, &str)> {
    let context_stack = i.context_stack;

    map_opt(identifier, move |name| {
        context_stack
            .iter()
            .rfind(|c| c.has(name))
            .map(|&c| (c, name))
    })(i)
}

fn named_constant(i: InputWithContext) -> ParseResult<Expr> {
    map_opt(name_in_context, |(ctx, name)| ctx.get_constant(name))(i)
}

fn function_name(i: InputWithContext) -> ParseResult<(&Context, &str)> {
    verify(name_in_context, |(ctx, name)| ctx.is_function(name))(i)
}

/// Nonempty, comma-separated list of expressions.
fn expr_list(i: InputWithContext) -> ParseResult<Vec<Expr>> {
    let (i, mut x) = expr(i)?;

    fold_many0(
        preceded(delimited(space0, char(','), space0), cut(expr)),
        move || vec![std::mem::take(&mut x)],
        |mut xs, x| {
            xs.push(x);
            xs
        },
    )(i)
}

fn function_application(i: InputWithContext) -> ParseResult<Expr> {
    map(
        pair(
            function_name,
            delimited(
                delimited(space0, cut(char('(')), space0),
                cut(expr_list),
                preceded(space0, cut(char(')'))),
            ),
        ),
        |((ctx, name), args)| ctx.apply(name, args),
    )(i)
}

/// If an identifier is found, [`cut`]s with [`ErrorKind::UnknownIdentifier`]
/// (the position where the identifier is found is reported);
/// otherwise, fails in the same manner as [`identifier`].
fn fail_unknown_identifier(i: InputWithContext) -> ParseResult<Expr> {
    let (i, name) = peek(identifier)(i)?;

    Err(Err::Failure(Error::unknown_identifier(i, name)))
}

/// Fails with [`ErrorKind::ExpectedExpr`].
fn fail_expr(i: InputWithContext) -> ParseResult<Expr> {
    Err(Err::Error(Error::expected_expr(i)))
}

fn expr_within_bars(i: InputWithContext) -> ParseResult<Expr> {
    let mut o = recognize(take_while(|c| c != '|'))(i.clone())?;
    let mut even_bars_taken = true;
    loop {
        let (rest, taken) = o;
        if even_bars_taken {
            if let Ok((_, x)) = all_consuming(expr)(taken.clone()) {
                return Ok((rest, x));
            }
        }
        if rest.input_len() == 0 {
            // Reached the end of input. All we can do is return a meaningful error.
            return expr(taken);
        }
        o = recognize(pair(take(taken.input_len() + 1), take_while(|c| c != '|')))(i.clone())?;
        even_bars_taken = !even_bars_taken;
    }
}

/// The inverse operation of [`cut`]; converts [`Err::Failure`] back to [`Err::Error`].
fn decut<I, O, E: ParseError<I>, F>(mut parser: F) -> impl FnMut(I) -> IResult<I, O, E>
where
    F: Parser<I, O, E>,
{
    move |input: I| match parser.parse(input) {
        Err(Err::Failure(e)) => Err(Err::Error(e)),
        rest => rest,
    }
}

fn primary_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();

    map(
        consumed(alt((
            decimal_constant,
            named_constant,
            function_application,
            fail_unknown_identifier,
            delimited(
                terminated(char('('), space0),
                cut(expr),
                preceded(space0, cut(char(')'))),
            ),
            map(
                delimited(
                    terminated(char('['), space0),
                    cut(expr_list),
                    preceded(space0, cut(char(']'))),
                ),
                |xs| Expr::nary(NaryOp::List, xs),
            ),
            map(
                delimited(
                    delimited(char('|'), peek(not(char('|'))), space0),
                    // Not an OR expression (unless it's called from the case below).
                    // So we can cut when no expression is found.
                    cut(expr_within_bars),
                    preceded(space0, cut(char('|'))),
                ),
                move |x| builtin.apply("abs", vec![x]),
            ),
            map(
                delimited(
                    terminated(char('|'), space0),
                    // Possibly an OR expression. We must not cut when no expression is found.
                    // The above case is called recursively, so we also need to cancel cut.
                    decut(expr_within_bars),
                    preceded(space0, cut(char('|'))),
                ),
                move |x| builtin.apply("abs", vec![x]),
            ),
            map(
                delimited(
                    terminated(char('⌈'), space0),
                    cut(expr),
                    preceded(space0, cut(char('⌉'))),
                ),
                move |x| builtin.apply("ceil", vec![x]),
            ),
            map(
                delimited(
                    terminated(char('⌊'), space0),
                    cut(expr),
                    preceded(space0, cut(char('⌋'))),
                ),
                move |x| builtin.apply("floor", vec![x]),
            ),
            fail_expr,
        ))),
        |(i, x)| x.with_source_range(i.source_range),
    )(i)
}

// ^ is right-associative; x^y^z is equivalent to x^(y^z).
fn power_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();

    map(
        pair(
            primary_expr,
            opt(pair(
                delimited(
                    space0,
                    alt((value("^^", tag("^^")), value("^", char('^')))),
                    space0,
                ),
                cut(unary_expr),
            )),
        ),
        move |(x, op_y)| match op_y {
            Some((op, y)) => {
                let range = x.source_range.start..y.source_range.end;
                builtin.apply(op, vec![x, y]).with_source_range(range)
            }
            _ => x,
        },
    )(i)
}

fn unary_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();

    alt((
        preceded(pair(char('+'), space0), cut(unary_expr)),
        map(
            consumed(separated_pair(
                alt((
                    value("~", char('~')),
                    value("-", one_of("-−")), // a hyphen-minus or a minus sign
                    value("!", one_of("!¬")),
                )),
                space0,
                cut(unary_expr),
            )),
            move |(i, (op, x))| builtin.apply(op, vec![x]).with_source_range(i.source_range),
        ),
        power_expr,
    ))(i)
}

fn multiplicative_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, mut x) = unary_expr(i)?;

    fold_many0(
        alt((
            // x * y
            // x / y
            pair(
                delimited(
                    space0,
                    alt((value("*", char('*')), value("/", char('/')))),
                    space0,
                ),
                cut(unary_expr),
            ),
            // 2x
            // x y
            pair(value("*", space0), power_expr),
        )),
        move || std::mem::take(&mut x),
        move |xs, (op, y)| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply(op, vec![xs, y]).with_source_range(range)
        },
    )(i)
}

fn additive_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, mut x) = multiplicative_expr(i)?;

    fold_many0(
        pair(
            delimited(
                space0,
                alt((
                    value("+", char('+')),
                    value("-", one_of("-−")), // a hyphen-minus or a minus sign
                )),
                space0,
            ),
            cut(multiplicative_expr),
        ),
        move || std::mem::take(&mut x),
        move |xs, (op, y)| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply(op, vec![xs, y]).with_source_range(range)
        },
    )(i)
}

// Relational operators can be chained: x op1 y op2 z is equivalent to x op1 y ∧ y op2 z.
fn relational_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, mut side) = additive_expr(i)?;

    map(
        fold_many0(
            pair(
                delimited(
                    space0,
                    alt((
                        value("=", char('=')),
                        value(">=", alt((tag(">="), tag("≥")))),
                        value(">", char('>')),
                        value("<=", alt((tag("<="), tag("≤")))),
                        value("<", char('<')),
                    )),
                    space0,
                ),
                cut(additive_expr),
            ),
            move || (vec![], vec![std::mem::take(&mut side)]),
            |(mut ops, mut sides), (op, side)| {
                ops.push(op);
                sides.push(side);
                (ops, sides)
            },
        ),
        move |(ops, sides)| {
            assert_eq!(sides.len(), ops.len() + 1);
            if sides.len() == 1 {
                sides.into_iter().next().unwrap()
            } else {
                ops.iter()
                    .zip(sides.windows(2))
                    .map(|(op, sides)| {
                        let range = sides[0].source_range.start..sides[1].source_range.end;
                        builtin.apply(op, sides.to_vec()).with_source_range(range)
                    })
                    .reduce(|xs, y| {
                        let range = xs.source_range.start..y.source_range.end;
                        builtin.apply("&&", vec![xs, y]).with_source_range(range)
                    })
                    .unwrap()
            }
        },
    )(i)
}

fn and_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, mut x) = relational_expr(i)?;

    fold_many0(
        preceded(
            delimited(space0, alt((tag("&&"), tag("∧"))), space0),
            cut(relational_expr),
        ),
        move || std::mem::take(&mut x),
        move |xs, y| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply("&&", vec![xs, y]).with_source_range(range)
        },
    )(i)
}

fn or_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, mut x) = and_expr(i)?;

    fold_many0(
        preceded(
            delimited(space0, alt((tag("||"), tag("∨"))), space0),
            cut(and_expr),
        ),
        move || std::mem::take(&mut x),
        move |xs, y| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply("||", vec![xs, y]).with_source_range(range)
        },
    )(i)
}

fn expr(i: InputWithContext) -> ParseResult<Expr> {
    or_expr(i)
}

/// Parses an expression.
pub fn parse_expr(source: &str, context_stack: &[&Context]) -> Result<Expr, String> {
    let i = InputWithContext::new(source, context_stack);
    match all_consuming(delimited(space0, expr, space0))(i.clone()).finish() {
        Ok((_, x)) => Ok(x),
        Err(e) => Err(convert_error(i, e)),
    }
}

pub fn format_error(source: &str, range: Range<usize>, message: &str) -> String {
    assert!(range.start <= range.end && range.end <= source.len());

    let offset = |substr: &str| {
        use nom::Offset;
        source.offset(substr)
    };

    let (line, source_line) = source
        .split('\n') // Do not use `.lines()` which ignores a final line ending.
        .enumerate()
        .take_while(|(_, line)| offset(*line) <= range.start)
        .last()
        .unwrap();
    let start_in_line = range.start - offset(source_line);
    let end_in_line = (range.end - offset(source_line)).min(source_line.len());
    let col = source_line[..start_in_line].chars().count();
    let n_cols = source_line[start_in_line..end_in_line].chars().count();
    let decoration = match n_cols {
        0 => "^".to_owned(),
        _ => "~".repeat(n_cols),
    };

    format!(
        r"
input:{}:{}: error: {}
{}
{:col$}{}
",
        line + 1,
        col + 1,
        message,
        source_line,
        "",
        decoration
    )
}

fn convert_error(input: InputWithContext, e: Error<InputWithContext>) -> String {
    use nom::Offset;

    let source = input.source;
    let offset = source.offset(e.input.source);
    let len = match e.kind {
        ErrorKind::UnknownIdentifier(name) => name.len(),
        _ => 0,
    };

    let message = match e.kind {
        ErrorKind::ExpectedChar(c) => format!("expected '{}'", c),
        ErrorKind::ExpectedEof => "unexpected input".to_owned(),
        ErrorKind::ExpectedExpr => "expected expression".to_owned(),
        ErrorKind::UnknownIdentifier(name) => format!("'{}' is not defined", name),
        _ => panic!("unexpected error kind"),
    };

    format_error(source, offset..offset + len, &message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Def, VarProps};

    #[test]
    fn parse_expr() {
        let ctx = Context::new()
            .def("a", Def::var("a", VarProps::default()))
            .def("b", Def::var("b", VarProps::default()))
            .def("k", Def::var("k", VarProps::default()))
            .def("n", Def::var("n", VarProps::default()))
            .def("z", Def::var("z", VarProps::default()));

        let test = |input, expected| {
            let f = super::parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            assert_eq!(format!("{}", f.dump_short()), expected);
        };

        test("false", "False");
        test("true", "True");
        test("e", "@");
        test("gamma", "@");
        test("γ", "@");
        test("pi", "@");
        test("π", "@");
        test("i", "(Complex 0 1)");
        test("[x, y, z]", "(List x y z)");
        test("|x|", "(Abs x)");
        test("||x| + y|", "(Abs (Add (Abs x) y))");
        test("|x + |y||", "(Abs (Add x (Abs y)))");
        test(
            "y = ||x|| || |||x|| + |||y|||| = y",
            "(Or (Eq y (Abs (Abs x))) (Eq (Abs (Add (Abs (Abs x)) (Abs (Abs (Abs y))))) y))",
        );
        test("⌈x⌉", "(Ceil x)");
        test("⌊x⌋", "(Floor x)");
        test("abs(x)", "(Abs x)");
        test("acos(x)", "(Acos x)");
        test("acosh(x)", "(Acosh x)");
        test("Ai(x)", "(AiryAi x)");
        test("Ai'(x)", "(AiryAiPrime x)");
        test("Bi(x)", "(AiryBi x)");
        test("Bi'(x)", "(AiryBiPrime x)");
        test("arg(x)", "(Arg x)");
        test("asin(x)", "(Asin x)");
        test("asinh(x)", "(Asinh x)");
        test("atan(x)", "(Atan x)");
        test("atanh(x)", "(Atanh x)");
        test("ceil(x)", "(Ceil x)");
        test("Chi(x)", "(Chi x)");
        test("Ci(x)", "(Ci x)");
        test("~x", "(Conj x)");
        test("cos(x)", "(Cos x)");
        test("cosh(x)", "(Cosh x)");
        test("psi(x)", "(Digamma x)");
        test("ψ(x)", "(Digamma x)");
        test("Ei(x)", "(Ei x)");
        test("E(x)", "(EllipticE x)");
        test("K(x)", "(EllipticK x)");
        test("erf(x)", "(Erf x)");
        test("erfc(x)", "(Erfc x)");
        test("erfi(x)", "(Erfi x)");
        test("exp(x)", "(Exp x)");
        test("floor(x)", "(Floor x)");
        test("C(x)", "(FresnelC x)");
        test("S(x)", "(FresnelS x)");
        test("Gamma(x)", "(Gamma x)");
        test("Γ(x)", "(Gamma x)");
        test("Im(x)", "(Im x)");
        test("erfinv(x)", "(InverseErf x)");
        test("erfcinv(x)", "(InverseErfc x)");
        test("li(x)", "(Li x)");
        test("ln(x)", "(Ln x)");
        test("lnGamma(x)", "(LnGamma x)");
        test("lnΓ(x)", "(LnGamma x)");
        test("-x", "(Neg x)"); // hyphen-minus
        test("−x", "(Neg x)"); // minus sign
        test("Re(x)", "(Re x)");
        test("Shi(x)", "(Shi x)");
        test("Si(x)", "(Si x)");
        test("sgn(x)", "(Sign x)");
        test("sign(x)", "(Sign x)");
        test("sin(x)", "(Sin x)");
        test("sinc(x)", "(Sinc x)");
        test("sinh(x)", "(Sinh x)");
        test("sqrt(x)", "(Sqrt x)");
        test("tan(x)", "(Tan x)");
        test("tanh(x)", "(Tanh x)");
        test("zeta(x)", "(Zeta x)");
        test("ζ(x)", "(Zeta x)");
        test("atan2(y, x)", "(Atan2 y x)");
        test("I(n, x)", "(BesselI n x)");
        test("J(n, x)", "(BesselJ n x)");
        test("K(n, x)", "(BesselK n x)");
        test("Y(n, x)", "(BesselY n x)");
        test("Gamma(a, x)", "(GammaInc a x)");
        test("Γ(a, x)", "(GammaInc a x)");
        test("W(x)", "(LambertW 0 x)");
        test("W(k, x)", "(LambertW k x)");
        test("log(b, x)", "(Log b x)");
        test("mod(x, y)", "(Mod x y)");
        test("gcd(x, y, z)", "(Gcd (Gcd x y) z)");
        test("lcm(x, y, z)", "(Lcm (Lcm x y) z)");
        test("max(x, y, z)", "(Max (Max x y) z)");
        test("min(x, y, z)", "(Min (Min x y) z)");
        test("if(x = 0, y, z)", "(IfThenElse (Boole (Eq x 0)) y z)");
        test("rankedMax([x, y, z], k)", "(RankedMax (List x y z) k)");
        test("rankedMin([x, y, z], k)", "(RankedMin (List x y z) k)");
        test("x ^ y ^ z", "(Pow x (Pow y z))");
        test("-x ^ -y", "(Neg (Pow x (Neg y)))");
        test("x ^^ y ^^ z", "(PowRational x (PowRational y z))");
        test("-x ^^ -y", "(Neg (PowRational x (Neg y)))");
        test("+x", "x");
        test("2x", "(Mul 2 x)");
        test("x y z", "(Mul (Mul x y) z)");
        test("x * y * z", "(Mul (Mul x y) z)");
        test("x / y / z", "(Div (Div x y) z)");
        test("x + y + z", "(Add (Add x y) z)");
        test("x - y - z", "(Sub (Sub x y) z)"); // hyphen-minus
        test("x − y − z", "(Sub (Sub x y) z)"); // minus sign
        test("x + y z", "(Add x (Mul y z))");
        test("(x + y) z", "(Mul (Add x y) z)");
        test("x = y", "(Eq x y)");
        test("x >= y", "(Ge x y)");
        test("x ≥ y", "(Ge x y)");
        test("x > y", "(Gt x y)");
        test("x <= y", "(Le x y)");
        test("x ≤ y", "(Le x y)");
        test("x < y", "(Lt x y)");
        test("x = y = z", "(And (Eq x y) (Eq y z))");
        test("!x", "(Not x)");
        test("¬x", "(Not x)");
        test("x && y", "(And x y)");
        test("x ∧ y", "(And x y)");
        test("x || y", "(Or x y)");
        test("x ∨ y", "(Or x y)");
        test("x = y && y = z", "(And (Eq x y) (Eq y z))");
        test("x = y || y = z", "(Or (Eq x y) (Eq y z))");
        test(
            "x = y || y = z && z = x",
            "(Or (Eq x y) (And (Eq y z) (Eq z x)))",
        );
        test(
            "(x = y || y = z) && z = x",
            "(And (Or (Eq x y) (Eq y z)) (Eq z x))",
        );
    }
}
