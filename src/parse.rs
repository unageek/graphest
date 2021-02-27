use crate::{
    ast::{BinaryOp, Form, FormKind, NaryOp, RelOp, Term, TermKind, UnaryOp},
    context::{Context, InputWithContext},
    interval_set::TupperIntervalSet,
};
use inari::{const_dec_interval, dec_interval, DecInterval};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, anychar, char, digit0, digit1, space0},
    combinator::{all_consuming, map, not, opt, peek, recognize, value, verify},
    error::VerboseError,
    multi::{fold_many0, fold_many1, many0},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Err as NomErr, IResult,
};
use std::collections::VecDeque;

type ParseResult<'a, O> = IResult<InputWithContext<'a>, O, VerboseError<InputWithContext<'a>>>;

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

fn keyword<'a>(
    kw: &'a str,
) -> impl FnMut(InputWithContext<'a>) -> ParseResult<'a, InputWithContext<'a>> {
    terminated(
        tag(kw),
        not(verify(peek(anychar), |c| {
            c.is_alphanumeric() || *c == '_' || *c == '\''
        })),
    )
}

fn variable(i: InputWithContext) -> ParseResult<&str> {
    let ctx = i.ctx;
    map(
        verify(
            recognize(pair(alpha1, many0(alphanumeric1))),
            move |s: &InputWithContext| !ctx.is_defined(s.i),
        ),
        |s: InputWithContext| s.i,
    )(i)
}

fn primary_term(i: InputWithContext) -> ParseResult<Term> {
    alt((
        map(decimal_literal, |s| {
            let s = ["[", s, ",", s, "]"].concat();
            let x = TupperIntervalSet::from(dec_interval!(&s).unwrap());
            Term::new(TermKind::Constant(Box::new(x)))
        }),
        map(keyword("e"), |_| {
            let x = TupperIntervalSet::from(DecInterval::E);
            Term::new(TermKind::Constant(Box::new(x)))
        }),
        map(alt((keyword("gamma"), keyword("γ"))), |_| {
            let x = TupperIntervalSet::from(const_dec_interval!(
                0.5772156649015328,
                0.5772156649015329
            ));
            Term::new(TermKind::Constant(Box::new(x)))
        }),
        map(alt((keyword("pi"), keyword("π"))), |_| {
            let x = TupperIntervalSet::from(DecInterval::PI);
            Term::new(TermKind::Constant(Box::new(x)))
        }),
        map(variable, |x| Term::new(TermKind::Var(x.into()))),
        delimited(
            terminated(char('('), space0),
            term,
            preceded(space0, char(')')),
        ),
        map(
            delimited(
                terminated(char('|'), space0),
                term,
                preceded(space0, char('|')),
            ),
            |x| Term::new(TermKind::Unary(UnaryOp::Abs, Box::new(x))),
        ),
        map(
            delimited(
                terminated(char('⌈'), space0),
                term,
                preceded(space0, char('⌉')),
            ),
            |x| Term::new(TermKind::Unary(UnaryOp::Ceil, Box::new(x))),
        ),
        map(
            delimited(
                terminated(char('⌊'), space0),
                term,
                preceded(space0, char('⌋')),
            ),
            |x| Term::new(TermKind::Unary(UnaryOp::Floor, Box::new(x))),
        ),
    ))(i)
}

fn fn1(i: InputWithContext) -> ParseResult<UnaryOp> {
    // `alt` takes a tuple with 21 elements at most.
    alt((
        value(UnaryOp::Acos, keyword("acos")),
        value(UnaryOp::Acosh, keyword("acosh")),
        value(UnaryOp::AiryAi, keyword("Ai")),
        value(UnaryOp::AiryAiPrime, keyword("Ai'")),
        value(UnaryOp::AiryBi, keyword("Bi")),
        value(UnaryOp::AiryBiPrime, keyword("Bi'")),
        value(UnaryOp::Asin, keyword("asin")),
        value(UnaryOp::Asinh, keyword("asinh")),
        value(UnaryOp::Atan, keyword("atan")),
        value(UnaryOp::Atanh, keyword("atanh")),
        value(UnaryOp::Ceil, keyword("ceil")),
        value(UnaryOp::Chi, keyword("Chi")),
        value(UnaryOp::Ci, keyword("Ci")),
        value(UnaryOp::Cos, keyword("cos")),
        value(UnaryOp::Cosh, keyword("cosh")),
        value(UnaryOp::Digamma, alt((keyword("psi"), keyword("ψ")))),
        value(UnaryOp::Ei, keyword("Ei")),
        value(UnaryOp::Erf, keyword("erf")),
        value(UnaryOp::Erfc, keyword("erfc")),
        value(UnaryOp::Erfi, keyword("erfi")),
        alt((
            value(UnaryOp::Exp, keyword("exp")),
            value(UnaryOp::Floor, keyword("floor")),
            value(UnaryOp::FresnelC, keyword("C")),
            value(UnaryOp::FresnelS, keyword("S")),
            value(UnaryOp::Gamma, alt((keyword("Gamma"), keyword("Γ")))),
            value(UnaryOp::Li, keyword("li")),
            value(UnaryOp::Ln, keyword("ln")),
            value(UnaryOp::Log10, keyword("log")),
            value(UnaryOp::Shi, keyword("Shi")),
            value(UnaryOp::Si, keyword("Si")),
            value(UnaryOp::Sign, keyword("sign")),
            value(UnaryOp::Sin, keyword("sin")),
            value(UnaryOp::Sinh, keyword("sinh")),
            value(UnaryOp::Sqrt, keyword("sqrt")),
            value(UnaryOp::Tan, keyword("tan")),
            value(UnaryOp::Tanh, keyword("tanh")),
        )),
    ))(i)
}

fn fn2(i: InputWithContext) -> ParseResult<BinaryOp> {
    alt((
        value(BinaryOp::Atan2, keyword("atan2")),
        value(BinaryOp::BesselI, keyword("I")),
        value(BinaryOp::BesselJ, keyword("J")),
        value(BinaryOp::BesselK, keyword("K")),
        value(BinaryOp::BesselY, keyword("Y")),
        value(BinaryOp::GammaInc, keyword("Gamma")),
        value(BinaryOp::GammaInc, keyword("Γ")),
        value(BinaryOp::Log, keyword("log")),
        value(BinaryOp::Mod, keyword("mod")),
    ))(i)
}

fn fn_flat(i: InputWithContext) -> ParseResult<BinaryOp> {
    alt((
        value(BinaryOp::Gcd, keyword("gcd")),
        value(BinaryOp::Lcm, keyword("lcm")),
        value(BinaryOp::Max, keyword("max")),
        value(BinaryOp::Min, keyword("min")),
    ))(i)
}

fn argument(i: InputWithContext) -> ParseResult<Term> {
    terminated(
        term,
        // Omit positional arguments.
        not(peek(pair(space0, char('=')))),
    )(i)
}

fn argument_list(i: InputWithContext) -> ParseResult<VecDeque<Term>> {
    let (i, x) = argument(i)?;

    let mut xs = VecDeque::new();
    xs.push_back(x);
    fold_many0(
        preceded(delimited(space0, char(','), space0), argument),
        xs,
        |mut xs, x| {
            xs.push_back(x);
            xs
        },
    )(i)
}

fn postfix_term(i: InputWithContext) -> ParseResult<Term> {
    alt((
        map(
            pair(
                fn1,
                delimited(
                    delimited(space0, char('('), space0),
                    term,
                    preceded(space0, char(')')),
                ),
            ),
            |(f, x)| Term::new(TermKind::Unary(f, Box::new(x))),
        ),
        map(
            pair(
                fn2,
                delimited(
                    delimited(space0, char('('), space0),
                    separated_pair(term, delimited(space0, char(','), space0), term),
                    preceded(space0, char(')')),
                ),
            ),
            |(f, (x, y))| Term::new(TermKind::Binary(f, Box::new(x), Box::new(y))),
        ),
        map(
            pair(
                fn_flat,
                delimited(
                    delimited(space0, char('('), space0),
                    argument_list,
                    preceded(space0, char(')')),
                ),
            ),
            |(f, mut xs)| {
                let head = xs.pop_front().unwrap();
                xs.into_iter().fold(head, |t, x| {
                    Term::new(TermKind::Binary(f, Box::new(t), Box::new(x)))
                })
            },
        ),
        map(
            pair(
                alt((
                    value(NaryOp::RankedMax, keyword("max")),
                    value(NaryOp::RankedMin, keyword("min")),
                )),
                delimited(
                    delimited(space0, char('('), space0),
                    separated_pair(
                        argument_list,
                        tuple((
                            space0,
                            char(','),
                            space0,
                            keyword("rank"),
                            space0,
                            char('='),
                            space0,
                        )),
                        term,
                    ),
                    preceded(space0, char(')')),
                ),
            ),
            |(f, (mut xs, n))| {
                xs.push_back(n);
                Term::new(TermKind::Nary(f, xs.into_iter().collect()))
            },
        ),
        primary_term,
    ))(i)
}

// ^ is right-associative: x^y^z is the same as x^(y^z).
fn power_term(i: InputWithContext) -> ParseResult<Term> {
    alt((
        map(
            separated_pair(
                postfix_term,
                delimited(space0, char('^'), space0),
                unary_term,
            ),
            |(x, y)| Term::new(TermKind::Binary(BinaryOp::Pow, Box::new(x), Box::new(y))),
        ),
        postfix_term,
    ))(i)
}

fn unary_term(i: InputWithContext) -> ParseResult<Term> {
    alt((
        preceded(pair(char('+'), space0), unary_term),
        map(preceded(pair(char('-'), space0), unary_term), |x| {
            Term::new(TermKind::Unary(UnaryOp::Neg, Box::new(x)))
        }),
        power_term,
    ))(i)
}

fn multiplicative_term(i: InputWithContext) -> ParseResult<Term> {
    let (i, x) = unary_term(i)?;

    fold_many0(
        alt((
            // x * y
            // x / y
            pair(
                delimited(
                    space0,
                    alt((
                        value(BinaryOp::Mul, char('*')),
                        value(BinaryOp::Div, char('/')),
                    )),
                    space0,
                ),
                unary_term,
            ),
            // x y
            pair(value(BinaryOp::Mul, space0), power_term),
        )),
        x,
        |xs, (op, y)| Term::new(TermKind::Binary(op, Box::new(xs), Box::new(y))),
    )(i)
}

fn additive_term(i: InputWithContext) -> ParseResult<Term> {
    let (i, x) = multiplicative_term(i)?;

    fold_many0(
        pair(
            delimited(
                space0,
                alt((
                    value(BinaryOp::Add, char('+')),
                    value(BinaryOp::Sub, char('-')),
                )),
                space0,
            ),
            multiplicative_term,
        ),
        x,
        |xs, (op, y)| Term::new(TermKind::Binary(op, Box::new(xs), Box::new(y))),
    )(i)
}

fn term(i: InputWithContext) -> ParseResult<Term> {
    additive_term(i)
}

// (In)equalities can be chained: x < y < z is the same as x < y && y < z.
fn equality(i: InputWithContext) -> ParseResult<Form> {
    // `acc` is a pair of `Vec<RelOp>` and `Vec<Term>` that store
    // lists of equality operators and their operands, respectively.
    // `acc.1.len() == acc.0.len() + 1` holds.
    let (i, acc) = map(term, |x| (vec![], vec![x]))(i)?;

    map(
        fold_many1(
            pair(
                delimited(
                    space0,
                    alt((
                        value(RelOp::Eq, char('=')),
                        value(RelOp::Ge, alt((tag(">="), tag("≥")))),
                        value(RelOp::Gt, char('>')),
                        value(RelOp::Le, alt((tag("<="), tag("≤")))),
                        value(RelOp::Lt, char('<')),
                    )),
                    space0,
                ),
                term,
            ),
            acc,
            |mut acc, (op, y)| {
                acc.0.push(op);
                acc.1.push(y);
                acc
            },
        ),
        |acc| {
            let op = acc.0[0];
            let x = acc.1[0].clone();
            let y = acc.1[1].clone();
            let mut f = Form::new(FormKind::Atomic(op, Box::new(x), Box::new(y)));
            for i in 1..acc.0.len() {
                let op = acc.0[i];
                let x = acc.1[i].clone();
                let y = acc.1[i + 1].clone();
                let f2 = Form::new(FormKind::Atomic(op, Box::new(x), Box::new(y)));
                f = Form::new(FormKind::And(Box::new(f), Box::new(f2)));
            }
            f
        },
    )(i)
}

fn primary_form(i: InputWithContext) -> ParseResult<Form> {
    alt((
        delimited(
            terminated(char('('), space0),
            form,
            preceded(space0, char(')')),
        ),
        equality,
    ))(i)
}

// Inputs like "!y < x" are allowed too.
fn not_form(i: InputWithContext) -> ParseResult<Form> {
    alt((
        map(preceded(pair(char('!'), space0), not_form), |x| {
            Form::new(FormKind::Not(Box::new(x)))
        }),
        primary_form,
    ))(i)
}

fn and_form(i: InputWithContext) -> ParseResult<Form> {
    let (i, x) = not_form(i)?;

    fold_many0(
        preceded(delimited(space0, tag("&&"), space0), not_form),
        x,
        |xs, y| Form::new(FormKind::And(Box::new(xs), Box::new(y))),
    )(i)
}

fn or_form(i: InputWithContext) -> ParseResult<Form> {
    let (i, x) = and_form(i)?;

    fold_many0(
        preceded(delimited(space0, tag("||"), space0), and_form),
        x,
        |xs, y| Form::new(FormKind::Or(Box::new(xs), Box::new(y))),
    )(i)
}

fn form(i: InputWithContext) -> ParseResult<Form> {
    or_form(i)
}

/// Parses a formula.
pub fn parse(i: &str, ctx: &Context) -> Result<Form, String> {
    let i = InputWithContext::new(i, ctx);
    match all_consuming(delimited(space0, form, space0))(i.clone()) {
        Ok((InputWithContext { i: "", ctx: _ }, x)) => Ok(x),
        Err(NomErr::Error(e)) | Err(NomErr::Failure(e)) => Err(convert_error(i, e)),
        _ => unreachable!(),
    }
}

// Copied from `nom::error::convert_error`.
#[allow(clippy::naive_bytecount)]
fn convert_error(input: InputWithContext, e: VerboseError<InputWithContext>) -> String {
    use nom::Offset;

    let input = input.i;
    let substring = e.errors.first().unwrap().0.i;
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
    let column_number = line.offset(substring) + 1;

    format!(
        "at line {line_number}:\n\
               {line}\n\
               {caret:>column$}\n\n",
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
    fn parse_term() {
        test_parse_term("|x|", "(Abs x)");
        test_parse_term("⌈x⌉", "(Ceil x)");
        test_parse_term("⌊x⌋", "(Floor x)");
        test_parse_term("acos(x)", "(Acos x)");
        test_parse_term("acosh(x)", "(Acosh x)");
        test_parse_term("Ai(x)", "(AiryAi x)");
        test_parse_term("Ai'(x)", "(AiryAiPrime x)");
        test_parse_term("Bi(x)", "(AiryBi x)");
        test_parse_term("Bi'(x)", "(AiryBiPrime x)");
        test_parse_term("asin(x)", "(Asin x)");
        test_parse_term("asinh(x)", "(Asinh x)");
        test_parse_term("atan(x)", "(Atan x)");
        test_parse_term("atanh(x)", "(Atanh x)");
        test_parse_term("ceil(x)", "(Ceil x)");
        test_parse_term("Chi(x)", "(Chi x)");
        test_parse_term("Ci(x)", "(Ci x)");
        test_parse_term("cos(x)", "(Cos x)");
        test_parse_term("cosh(x)", "(Cosh x)");
        test_parse_term("psi(x)", "(Digamma x)");
        test_parse_term("ψ(x)", "(Digamma x)");
        test_parse_term("Ei(x)", "(Ei x)");
        test_parse_term("erf(x)", "(Erf x)");
        test_parse_term("erfc(x)", "(Erfc x)");
        test_parse_term("erfi(x)", "(Erfi x)");
        test_parse_term("exp(x)", "(Exp x)");
        test_parse_term("floor(x)", "(Floor x)");
        test_parse_term("C(x)", "(FresnelC x)");
        test_parse_term("S(x)", "(FresnelS x)");
        test_parse_term("Gamma(x)", "(Gamma x)");
        test_parse_term("Γ(x)", "(Gamma x)");
        test_parse_term("li(x)", "(Li x)");
        test_parse_term("ln(x)", "(Ln x)");
        test_parse_term("log(x)", "(Log10 x)");
        test_parse_term("Shi(x)", "(Shi x)");
        test_parse_term("Si(x)", "(Si x)");
        test_parse_term("sign(x)", "(Sign x)");
        test_parse_term("sin(x)", "(Sin x)");
        test_parse_term("sinh(x)", "(Sinh x)");
        test_parse_term("sqrt(x)", "(Sqrt x)");
        test_parse_term("tan(x)", "(Tan x)");
        test_parse_term("tanh(x)", "(Tanh x)");
        test_parse_term("atan2(y, x)", "(Atan2 y x)");
        test_parse_term("I(n, x)", "(BesselI n x)");
        test_parse_term("J(n, x)", "(BesselJ n x)");
        test_parse_term("K(n, x)", "(BesselK n x)");
        test_parse_term("Y(n, x)", "(BesselY n x)");
        test_parse_term("Gamma(a, x)", "(GammaInc a x)");
        test_parse_term("Γ(a, x)", "(GammaInc a x)");
        test_parse_term("log(b, x)", "(Log b x)");
        test_parse_term("mod(x, y)", "(Mod x y)");
        test_parse_term("gcd(x, y, z)", "(Gcd (Gcd x y) z)");
        test_parse_term("lcm(x, y, z)", "(Lcm (Lcm x y) z)");
        test_parse_term("max(x, y, z)", "(Max (Max x y) z)");
        test_parse_term("min(x, y, z)", "(Min (Min x y) z)");
        test_parse_term("max(x, y, z, rank=k)", "(RankedMax x y z k)");
        test_parse_term("min(x, y, z, rank=k)", "(RankedMin x y z k)");
        test_parse_term("x ^ y ^ z", "(Pow x (Pow y z))");
        test_parse_term("-x ^ -y", "(Neg (Pow x (Neg y)))");
        test_parse_term("+x", "x");
        test_parse_term("-x", "(Neg x)");
        test_parse_term("x y z", "(Mul (Mul x y) z)");
        test_parse_term("x * y * z", "(Mul (Mul x y) z)");
        test_parse_term("x / y / z", "(Div (Div x y) z)");
        test_parse_term("x + y + z", "(Add (Add x y) z)");
        test_parse_term("x - y - z", "(Sub (Sub x y) z)");
        test_parse_term("x + y z", "(Add x (Mul y z))");
        test_parse_term("(x + y) z", "(Mul (Add x y) z)");
    }

    fn test_parse_term(input: &str, expected: &str) {
        let ctx = Context::new();
        let f = super::parse(&format!("{} = 0", input), &ctx).unwrap();
        assert_eq!(
            format!("(Eq {} {{...}})", expected),
            format!("{}", f.dump_structure())
        );
    }

    #[test]
    fn parse_forms() {
        test_parse_form("x = y", "(Eq x y)");
        test_parse_form("x >= y", "(Ge x y)");
        test_parse_form("x ≥ y", "(Ge x y)");
        test_parse_form("x > y", "(Gt x y)");
        test_parse_form("x <= y", "(Le x y)");
        test_parse_form("x ≤ y", "(Le x y)");
        test_parse_form("x < y", "(Lt x y)");
        test_parse_form("x = y = z", "(And (Eq x y) (Eq y z))");
        test_parse_form("!x = y", "(Not (Eq x y))");
        test_parse_form("x = y && y = z", "(And (Eq x y) (Eq y z))");
        test_parse_form("x = y || y = z", "(Or (Eq x y) (Eq y z))");
        test_parse_form(
            "x = y || y = z && z = x",
            "(Or (Eq x y) (And (Eq y z) (Eq z x)))",
        );
        test_parse_form(
            "(x = y || y = z) && z = x",
            "(And (Or (Eq x y) (Eq y z)) (Eq z x))",
        );
    }

    fn test_parse_form(input: &str, expected: &str) {
        let ctx = Context::new();
        let f = super::parse(input, &ctx).unwrap();
        assert_eq!(expected, format!("{}", f.dump_structure()));
    }
}
