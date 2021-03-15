use crate::{
    ast::{Expr, ExprKind},
    context::{Context, InputWithContext},
    interval_set::TupperIntervalSet,
};
use inari::dec_interval;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit0, digit1, satisfy, space0},
    combinator::{all_consuming, cut, map, map_opt, not, opt, peek, recognize, value},
    error::VerboseError,
    multi::{fold_many0, many0_count},
    sequence::{delimited, pair, preceded, separated_pair, terminated},
    Err as NomErr, IResult,
};

type ParseResult<'a, O> = IResult<InputWithContext<'a>, O, VerboseError<InputWithContext<'a>>>;

fn identifier_head(i: InputWithContext) -> ParseResult<char> {
    satisfy(|c| c.is_alphabetic())(i)
}

fn identifier_tail(i: InputWithContext) -> ParseResult<&str> {
    map(
        recognize(many0_count(satisfy(|c| {
            c.is_alphanumeric() || c == '_' || c == '\''
        }))),
        |s: InputWithContext| s.i,
    )(i)
}

fn identifier(i: InputWithContext) -> ParseResult<&str> {
    map(recognize(pair(identifier_head, identifier_tail)), |s| s.i)(i)
}

fn decimal_literal(i: InputWithContext) -> ParseResult<&str> {
    map(
        alt((
            // "12", "12." or "12.3"
            recognize(pair(digit1, opt(pair(char('.'), digit0)))),
            // ".3"
            recognize(pair(char('.'), digit1)),
        )),
        |s: InputWithContext| s.i,
    )(i)
}

fn decimal_constant(i: InputWithContext) -> ParseResult<Expr> {
    map(decimal_literal, |s| {
        let s = ["[", s, ",", s, "]"].concat();
        let x = TupperIntervalSet::from(dec_interval!(&s).unwrap());
        Expr::new(ExprKind::Constant(Box::new(x)))
    })(i)
}

fn named_constant(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
    map_opt(identifier, move |s| ctx.get_constant(s))(i)
}

/// Nonempty, comma-separated list of expressions.
fn expr_list(i: InputWithContext) -> ParseResult<Vec<Expr>> {
    let (i, x) = expr(i)?;

    fold_many0(
        preceded(delimited(space0, char(','), space0), expr),
        vec![x],
        |mut xs, x| {
            xs.push(x);
            xs
        },
    )(i)
}

fn function_application(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;

    map_opt(
        pair(
            identifier,
            delimited(
                delimited(space0, char('('), space0),
                expr_list,
                preceded(space0, cut(char(')'))),
            ),
        ),
        move |(s, args)| ctx.apply(s, args),
    )(i)
}

fn variable(i: InputWithContext) -> ParseResult<Expr> {
    map(identifier, |s| Expr::new(ExprKind::Var(s.into())))(i)
}

fn primary_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;

    alt((
        decimal_constant,
        named_constant,
        function_application,
        variable,
        delimited(
            terminated(char('('), space0),
            expr,
            preceded(space0, cut(char(')'))),
        ),
        map(
            delimited(
                terminated(char('['), space0),
                expr_list,
                preceded(space0, cut(char(']'))),
            ),
            |xs| Expr::new(ExprKind::List(xs)),
        ),
        map_opt(
            delimited(
                terminated(terminated(char('|'), not(peek(char('|')))), space0),
                expr,
                preceded(space0, char('|')),
            ),
            move |x| ctx.apply("abs", vec![x]),
        ),
        map_opt(
            delimited(
                terminated(char('⌈'), space0),
                expr,
                preceded(space0, cut(char('⌉'))),
            ),
            move |x| ctx.apply("ceil", vec![x]),
        ),
        map_opt(
            delimited(
                terminated(char('⌊'), space0),
                expr,
                preceded(space0, cut(char('⌋'))),
            ),
            move |x| ctx.apply("floor", vec![x]),
        ),
    ))(i)
}

// ^ is right-associative: x^y^z is the same as x^(y^z).
fn power_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;

    alt((
        map(
            separated_pair(
                primary_expr,
                delimited(space0, char('^'), space0),
                cut(unary_expr),
            ),
            move |(x, y)| ctx.apply("^", vec![x, y]).unwrap(),
        ),
        primary_expr,
    ))(i)
}

fn unary_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;

    alt((
        preceded(pair(char('+'), space0), cut(unary_expr)),
        map(
            separated_pair(
                alt((value("-", char('-')), value("!", char('!')))),
                space0,
                cut(unary_expr),
            ),
            move |(op, x)| ctx.apply(op, vec![x]).unwrap(),
        ),
        power_expr,
    ))(i)
}

fn multiplicative_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
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
        x,
        move |xs, (op, y)| ctx.apply(op, vec![xs, y]).unwrap(),
    )(i)
}

fn additive_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
    let (i, x) = multiplicative_expr(i)?;

    fold_many0(
        pair(
            delimited(
                space0,
                alt((value("+", char('+')), value("-", char('-')))),
                space0,
            ),
            cut(multiplicative_expr),
        ),
        x,
        move |xs, (op, y)| ctx.apply(op, vec![xs, y]).unwrap(),
    )(i)
}

// Relational operators can be chained: x < y < z is the same as x < y && y < z.
fn relational_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
    let (i, (ops, xs)) = map(additive_expr, |x| (vec![], vec![x]))(i)?;

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
            (ops, xs),
            |(mut ops, mut xs), (op, y)| {
                ops.push(op);
                xs.push(y);
                (ops, xs)
            },
        ),
        move |(ops, xs)| {
            assert_eq!(xs.len(), ops.len() + 1);
            if ops.is_empty() {
                xs[0].clone()
            } else {
                let op = ops[0];
                let x = xs[0].clone();
                let y = xs[1].clone();
                let mut t = ctx.apply(op, vec![x, y]).unwrap();
                for i in 1..ops.len() {
                    let op = ops[i];
                    let x = xs[i].clone();
                    let y = xs[i + 1].clone();
                    let t2 = ctx.apply(op, vec![x, y]).unwrap();
                    t = ctx.apply("&&", vec![t, t2]).unwrap();
                }
                t
            }
        },
    )(i)
}

fn and_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
    let (i, x) = relational_expr(i)?;

    fold_many0(
        preceded(delimited(space0, tag("&&"), space0), cut(relational_expr)),
        x,
        move |xs, y| ctx.apply("&&", vec![xs, y]).unwrap(),
    )(i)
}

fn or_expr(i: InputWithContext) -> ParseResult<Expr> {
    let ctx = i.ctx;
    let (i, x) = and_expr(i)?;

    fold_many0(
        preceded(delimited(space0, tag("||"), space0), cut(and_expr)),
        x,
        move |xs, y| ctx.apply("||", vec![xs, y]).unwrap(),
    )(i)
}

fn expr(i: InputWithContext) -> ParseResult<Expr> {
    or_expr(i)
}

/// Parses an expression.
pub fn parse_expr(i: &str, ctx: &Context) -> Result<Expr, String> {
    let i = InputWithContext::new(i, ctx);
    match all_consuming(delimited(space0, expr, space0))(i.clone()) {
        Ok((InputWithContext { i: "", ctx: _ }, x)) => Ok(x),
        Err(NomErr::Error(e)) | Err(NomErr::Failure(e)) => Err(convert_error(i, e)),
        _ => unreachable!(),
    }
}

// Based on `nom::error::convert_error`.
#[allow(clippy::naive_bytecount)]
fn convert_error(input: InputWithContext, e: VerboseError<InputWithContext>) -> String {
    use nom::Offset;

    let input = input.i;
    let substring = e.errors.first().unwrap().0.i;
    // Skip leading spaces for readability.
    let ws_chars = &[' ', '\t'][..];
    let substring = substring.trim_start_matches(ws_chars);
    let message = match substring.split(ws_chars).next() {
        Some(word) if !word.is_empty() => format!("unexpected input around '{}'", word),
        _ => "unexpected end of input".into(),
    };
    let offset = input.offset(substring);

    let prefix = &input.as_bytes()[..offset];

    // Count the number of newlines in the first `offset` bytes of input
    let line_number = prefix.iter().filter(|&&b| b == b'\n').count() + 1;

    // Find the line that includes the subslice:
    // Find the *last* newline before the substring starts
    let line_begin = prefix
        .iter()
        .rev()
        .position(|&b| b == b'\n')
        .map(|pos| offset - pos)
        .unwrap_or(0);

    // Find the full line after that newline
    let line = input[line_begin..]
        .lines()
        .next()
        .unwrap_or(&input[line_begin..])
        .trim_end();

    // The (1-indexed) column number is the offset of our substring into that line
    let column_number = line[..line.offset(substring)].chars().count() + 1;

    format!(
        "{message} at line {line_number}:\n\
               {line}\n\
               {caret:>column$}\n\n",
        message = message,
        line_number = line_number,
        line = line,
        caret = '^',
        column = column_number,
    )
}

#[cfg(test)]
mod tests {
    use crate::context::Context;

    #[test]
    fn parse_expr() {
        test_parse_expr("e", "@");
        test_parse_expr("gamma", "@");
        test_parse_expr("γ", "@");
        test_parse_expr("pi", "@");
        test_parse_expr("π", "@");
        test_parse_expr("[x, y, z]", "(List x y z)");
        test_parse_expr("|x|", "(Abs x)");
        test_parse_expr("|(|x| + y)|", "(Abs (Add (Abs x) y))");
        test_parse_expr("⌈x⌉", "(Ceil x)");
        test_parse_expr("⌊x⌋", "(Floor x)");
        test_parse_expr("abs(x)", "(Abs x)");
        test_parse_expr("acos(x)", "(Acos x)");
        test_parse_expr("acosh(x)", "(Acosh x)");
        test_parse_expr("Ai(x)", "(AiryAi x)");
        test_parse_expr("Ai'(x)", "(AiryAiPrime x)");
        test_parse_expr("Bi(x)", "(AiryBi x)");
        test_parse_expr("Bi'(x)", "(AiryBiPrime x)");
        test_parse_expr("asin(x)", "(Asin x)");
        test_parse_expr("asinh(x)", "(Asinh x)");
        test_parse_expr("atan(x)", "(Atan x)");
        test_parse_expr("atanh(x)", "(Atanh x)");
        test_parse_expr("ceil(x)", "(Ceil x)");
        test_parse_expr("Chi(x)", "(Chi x)");
        test_parse_expr("Ci(x)", "(Ci x)");
        test_parse_expr("cos(x)", "(Cos x)");
        test_parse_expr("cosh(x)", "(Cosh x)");
        test_parse_expr("psi(x)", "(Digamma x)");
        test_parse_expr("ψ(x)", "(Digamma x)");
        test_parse_expr("Ei(x)", "(Ei x)");
        test_parse_expr("erf(x)", "(Erf x)");
        test_parse_expr("erfc(x)", "(Erfc x)");
        test_parse_expr("erfi(x)", "(Erfi x)");
        test_parse_expr("exp(x)", "(Exp x)");
        test_parse_expr("floor(x)", "(Floor x)");
        test_parse_expr("C(x)", "(FresnelC x)");
        test_parse_expr("S(x)", "(FresnelS x)");
        test_parse_expr("Gamma(x)", "(Gamma x)");
        test_parse_expr("Γ(x)", "(Gamma x)");
        test_parse_expr("li(x)", "(Li x)");
        test_parse_expr("ln(x)", "(Ln x)");
        test_parse_expr("log(x)", "(Log10 x)");
        test_parse_expr("Shi(x)", "(Shi x)");
        test_parse_expr("Si(x)", "(Si x)");
        test_parse_expr("sign(x)", "(Sign x)");
        test_parse_expr("sin(x)", "(Sin x)");
        test_parse_expr("sinh(x)", "(Sinh x)");
        test_parse_expr("sqrt(x)", "(Sqrt x)");
        test_parse_expr("tan(x)", "(Tan x)");
        test_parse_expr("tanh(x)", "(Tanh x)");
        test_parse_expr("atan2(y, x)", "(Atan2 y x)");
        test_parse_expr("I(n, x)", "(BesselI n x)");
        test_parse_expr("J(n, x)", "(BesselJ n x)");
        test_parse_expr("K(n, x)", "(BesselK n x)");
        test_parse_expr("Y(n, x)", "(BesselY n x)");
        test_parse_expr("Gamma(a, x)", "(GammaInc a x)");
        test_parse_expr("Γ(a, x)", "(GammaInc a x)");
        test_parse_expr("log(b, x)", "(Log b x)");
        test_parse_expr("mod(x, y)", "(Mod x y)");
        test_parse_expr("gcd(x, y, z)", "(Gcd (Gcd x y) z)");
        test_parse_expr("lcm(x, y, z)", "(Lcm (Lcm x y) z)");
        test_parse_expr("max(x, y, z)", "(Max (Max x y) z)");
        test_parse_expr("min(x, y, z)", "(Min (Min x y) z)");
        test_parse_expr("ranked_max([x, y, z], k)", "(RankedMax (List x y z) k)");
        test_parse_expr("ranked_min([x, y, z], k)", "(RankedMin (List x y z) k)");
        test_parse_expr("x ^ y ^ z", "(Pow x (Pow y z))");
        test_parse_expr("-x ^ -y", "(Neg (Pow x (Neg y)))");
        test_parse_expr("+x", "x");
        test_parse_expr("-x", "(Neg x)");
        test_parse_expr("2x", "(Mul @ x)");
        test_parse_expr("x y z", "(Mul (Mul x y) z)");
        test_parse_expr("x * y * z", "(Mul (Mul x y) z)");
        test_parse_expr("x / y / z", "(Div (Div x y) z)");
        test_parse_expr("x + y + z", "(Add (Add x y) z)");
        test_parse_expr("x - y - z", "(Sub (Sub x y) z)");
        test_parse_expr("x + y z", "(Add x (Mul y z))");
        test_parse_expr("(x + y) z", "(Mul (Add x y) z)");
        test_parse_expr("x = y", "(Eq x y)");
        test_parse_expr("x >= y", "(Ge x y)");
        test_parse_expr("x ≥ y", "(Ge x y)");
        test_parse_expr("x > y", "(Gt x y)");
        test_parse_expr("x <= y", "(Le x y)");
        test_parse_expr("x ≤ y", "(Le x y)");
        test_parse_expr("x < y", "(Lt x y)");
        test_parse_expr("x = y = z", "(And (Eq x y) (Eq y z))");
        test_parse_expr("!(x = y)", "(Not (Eq x y))");
        test_parse_expr("x = y && y = z", "(And (Eq x y) (Eq y z))");
        test_parse_expr("x = y || y = z", "(Or (Eq x y) (Eq y z))");
        test_parse_expr(
            "x = y || y = z && z = x",
            "(Or (Eq x y) (And (Eq y z) (Eq z x)))",
        );
        test_parse_expr(
            "(x = y || y = z) && z = x",
            "(And (Or (Eq x y) (Eq y z)) (Eq z x))",
        );
    }

    fn test_parse_expr(input: &str, expected: &str) {
        let f = super::parse_expr(input, Context::builtin_context()).unwrap();
        assert_eq!(format!("{}", f.dump_structure()), expected);
    }
}
