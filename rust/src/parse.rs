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
        all_consuming, consumed, cut, fail, map, map_opt, not, opt, peek, recognize, value, verify,
    },
    error::{context, ErrorKind, VerboseError, VerboseErrorKind},
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    Finish, IResult, InputLength,
};
use rug::{Integer, Rational};
use std::ops::Range;

type ParseResult<'a, O> = IResult<InputWithContext<'a>, O, VerboseError<InputWithContext<'a>>>;

// Based on `inari::parse::parse_dec_float`.
fn parse_decimal(mant: &str) -> Option<Rational> {
    fn pow(base: u32, exp: i32) -> Rational {
        let i = Integer::from(Integer::u_pow_u(base, exp.abs() as u32));
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
    let (i, x) = expr(i)?;

    fold_many0(
        preceded(delimited(space0, char(','), space0), cut(expr)),
        move || vec![x.clone()],
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

fn shortest_expr_within_bars(i: InputWithContext) -> ParseResult<Expr> {
    let mut min_len = 0;

    loop {
        // In kth iteration, the next line tries to take the longest input
        // that contains exactly 2(k - 1) bars.
        let (rest, taken) = recognize(pair(take(min_len), take_while(|c| c != '|')))(i.clone())?;
        min_len = taken.input_len() + 1;

        if let Ok((_, x)) = all_consuming(expr)(taken.clone()) {
            return Ok((rest, x));
        } else if rest.input_len() == 0 {
            // Reached the end of input. All we can do is return a meaningful error.
            return expr(taken);
        }

        let (_, taken) = recognize(pair(take(min_len), take_while(|c| c != '|')))(i.clone())?;
        min_len = taken.input_len() + 1;
    }
}

fn primary_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();

    map(
        consumed(alt((
            decimal_constant,
            named_constant,
            function_application,
            map(identifier, |_| Expr::error()),
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
                    // Certainly not an OR expression. We can cut when no expression is found.
                    cut(shortest_expr_within_bars),
                    preceded(space0, cut(char('|'))),
                ),
                move |x| builtin.apply("abs", vec![x]),
            ),
            map(
                delimited(
                    terminated(char('|'), space0),
                    // Possibly an OR expression. We cannot cut when no expression is found.
                    shortest_expr_within_bars,
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
            context("an expression", fail),
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
            opt(preceded(
                delimited(space0, char('^'), space0),
                cut(unary_expr),
            )),
        ),
        move |(x, y)| match y {
            Some(y) => {
                let range = x.source_range.start..y.source_range.end;
                builtin.apply("^", vec![x, y]).with_source_range(range)
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
    let (i, x) = unary_expr(i)?;

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
        move || x.clone(),
        move |xs, (op, y)| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply(op, vec![xs, y]).with_source_range(range)
        },
    )(i)
}

fn additive_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, x) = multiplicative_expr(i)?;

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
        move || x.clone(),
        move |xs, (op, y)| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply(op, vec![xs, y]).with_source_range(range)
        },
    )(i)
}

// Relational operators can be chained: x op1 y op2 z is equivalent to x op1 y ∧ y op2 z.
fn relational_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, side) = additive_expr(i)?;
    let ops = vec![];
    let sides = vec![side];

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
            move || (ops.clone(), sides.clone()),
            |(mut ops, mut sides), (op, side)| {
                ops.push(op);
                sides.push(side);
                (ops, sides)
            },
        ),
        move |(ops, sides)| {
            assert_eq!(sides.len(), ops.len() + 1);
            if sides.len() == 1 {
                sides[0].clone()
            } else {
                let mut it = ops.iter().zip(sides.windows(2)).map(|(op, sides)| {
                    let range = sides[0].source_range.start..sides[1].source_range.end;
                    builtin.apply(op, sides.to_vec()).with_source_range(range)
                });
                let x = it.next().unwrap();
                it.fold(x, |xs, y| {
                    let range = xs.source_range.start..y.source_range.end;
                    builtin.apply("&&", vec![xs, y]).with_source_range(range)
                })
            }
        },
    )(i)
}

fn and_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, x) = relational_expr(i)?;

    fold_many0(
        preceded(
            delimited(space0, alt((tag("&&"), tag("∧"))), space0),
            cut(relational_expr),
        ),
        move || x.clone(),
        move |xs, y| {
            let range = xs.source_range.start..y.source_range.end;
            builtin.apply("&&", vec![xs, y]).with_source_range(range)
        },
    )(i)
}

fn or_expr(i: InputWithContext) -> ParseResult<Expr> {
    let builtin = i.context_stack.first().unwrap();
    let (i, x) = and_expr(i)?;

    fold_many0(
        preceded(
            delimited(space0, alt((tag("||"), tag("∨"))), space0),
            cut(and_expr),
        ),
        move || x.clone(),
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

fn convert_error(input: InputWithContext, e: VerboseError<InputWithContext>) -> String {
    use nom::Offset;

    let error = e
        .errors
        .iter()
        .find(|e| matches!(e.1, VerboseErrorKind::Context(_)))
        .or_else(|| e.errors.first())
        .unwrap();
    let message = match error.1 {
        VerboseErrorKind::Context(what) => format!("expected {}", what),
        VerboseErrorKind::Char(c) => format!("expected '{}'", c),
        VerboseErrorKind::Nom(ErrorKind::Eof) => "expected end of input".to_owned(),
        _ => panic!(),
    };

    let source = input.source;
    let substring = error.0.source;
    let offset = source.offset(substring);
    format_error(source, offset..offset, &message)
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
        test("li(x)", "(Li x)");
        test("ln(x)", "(Ln x)");
        test("-x", "(Neg x)"); // hyphen-minus
        test("−x", "(Neg x)"); // minus sign
        test("Re(x)", "(Re x)");
        test("Shi(x)", "(Shi x)");
        test("Si(x)", "(Si x)");
        test("sgn(x)", "(Sign x)");
        test("sign(x)", "(Sign x)");
        test("sin(x)", "(Sin x)");
        test("sinh(x)", "(Sinh x)");
        test("sqrt(x)", "(Sqrt x)");
        test("tan(x)", "(Tan x)");
        test("tanh(x)", "(Tanh x)");
        test("atan2(y, x)", "(Atan2 y x)");
        test("I(n, x)", "(BesselI n x)");
        test("J(n, x)", "(BesselJ n x)");
        test("K(n, x)", "(BesselK n x)");
        test("Y(n, x)", "(BesselY n x)");
        test("Gamma(a, x)", "(GammaInc a x)");
        test("Γ(a, x)", "(GammaInc a x)");
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
