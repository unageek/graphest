use crate::{ast::*, interval_set::*, rel::*};
use inari::*;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit0, digit1, one_of, space0},
    combinator::{all_consuming, map, map_res, opt, recognize, value},
    error::VerboseError,
    multi::fold_many0,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Err as NomErr, IResult,
};

type ParseResult<'a, O> = IResult<&'a str, O, VerboseError<&'a str>>;

fn integer_literal(s: &str) -> ParseResult<i32> {
    map_res(recognize(pair(opt(one_of("+-")), digit1)), |i: &str| {
        i.parse::<i32>()
    })(s)
}

fn decimal_literal(i: &str) -> ParseResult<&str> {
    alt((
        recognize(pair(digit1, opt(pair(char('.'), digit0)))),
        recognize(pair(char('.'), digit1)),
    ))(i)
}

fn primary_expr(i: &str) -> ParseResult<Expr> {
    alt((
        map(decimal_literal, |s| {
            let s = ["[", s, ",", s, "]"].concat();
            let x = TupperIntervalSet::from(dec_interval!(&s).unwrap());
            Expr::new(ExprKind::Constant(box x))
        }),
        map(tag("pi"), |_| {
            let x = TupperIntervalSet::from(DecoratedInterval::PI);
            Expr::new(ExprKind::Constant(box x))
        }),
        map(char('e'), |_| {
            let x = TupperIntervalSet::from(DecoratedInterval::E);
            Expr::new(ExprKind::Constant(box x))
        }),
        value(Expr::new(ExprKind::X), char('x')),
        value(Expr::new(ExprKind::Y), char('y')),
        delimited(
            terminated(char('('), space0),
            expr,
            preceded(space0, char(')')),
        ),
    ))(i)
}

fn fn1(i: &str) -> ParseResult<UnaryOp> {
    // alt is limited to 21 choices.
    alt((
        value(UnaryOp::Acosh, tag("acosh")),
        value(UnaryOp::Asinh, tag("asinh")),
        value(UnaryOp::Atanh, tag("atanh")),
        value(UnaryOp::Exp10, tag("exp10")),
        value(UnaryOp::Floor, tag("floor")),
        value(UnaryOp::Log10, tag("log10")),
        value(UnaryOp::Acos, tag("acos")),
        value(UnaryOp::Asin, tag("asin")),
        value(UnaryOp::Atan, tag("atan")),
        value(UnaryOp::Ceil, tag("ceil")),
        value(UnaryOp::Cosh, tag("cosh")),
        value(UnaryOp::Exp2, tag("exp2")),
        value(UnaryOp::Log2, tag("log2")),
        value(UnaryOp::Sinh, tag("sinh")),
        value(UnaryOp::Sqrt, tag("sqrt")),
        value(UnaryOp::Tanh, tag("tanh")),
        alt((
            value(UnaryOp::Abs, tag("abs")),
            value(UnaryOp::Cos, tag("cos")),
            value(UnaryOp::Exp, tag("exp")),
            value(UnaryOp::Log, tag("log")),
            value(UnaryOp::Sign, tag("sgn")),
            value(UnaryOp::Sin, tag("sin")),
            value(UnaryOp::Tan, tag("tan")),
        )),
    ))(i)
}

fn fn2(i: &str) -> ParseResult<BinaryOp> {
    alt((
        value(BinaryOp::Atan2, tag("atan2")),
        value(BinaryOp::Max, tag("max")),
        value(BinaryOp::Min, tag("min")),
        value(BinaryOp::Mod, tag("mod")),
    ))(i)
}

fn postfix_expr(i: &str) -> ParseResult<Expr> {
    alt((
        map(
            pair(
                fn1,
                delimited(
                    terminated(char('('), space0),
                    expr,
                    preceded(space0, char(')')),
                ),
            ),
            move |(f, x)| Expr::new(ExprKind::Unary(f, Box::new(x))),
        ),
        map(
            pair(
                fn2,
                delimited(
                    terminated(char('('), space0),
                    separated_pair(expr, delimited(space0, char(','), space0), expr),
                    preceded(space0, char(')')),
                ),
            ),
            move |(f, (x, y))| Expr::new(ExprKind::Binary(f, Box::new(x), Box::new(y))),
        ),
        primary_expr,
    ))(i)
}

// Repetition like x^2^3 (== x^(2^3)) is not permitted at the moment.
fn power_expr(i: &str) -> ParseResult<Expr> {
    alt((
        map(
            tuple((
                postfix_expr,
                delimited(space0, char('^'), space0),
                integer_literal,
            )),
            move |(x, _, y)| Expr::new(ExprKind::Pown(Box::new(x), y)),
        ),
        postfix_expr,
    ))(i)
}

fn unary_expr(i: &str) -> ParseResult<Expr> {
    alt((
        map(
            separated_pair(value(UnaryOp::Neg, char('-')), space0, unary_expr),
            |(op, x)| Expr::new(ExprKind::Unary(op, Box::new(x))),
        ),
        power_expr,
    ))(i)
}

fn multiplicative_expr(i: &str) -> ParseResult<Expr> {
    let (i, x) = unary_expr(i)?;

    fold_many0(
        pair(
            delimited(
                space0,
                alt((
                    value(BinaryOp::Mul, char('*')),
                    value(BinaryOp::Div, char('/')),
                )),
                space0,
            ),
            unary_expr,
        ),
        x,
        move |xs, (op, y)| Expr::new(ExprKind::Binary(op, Box::new(xs), Box::new(y))),
    )(i)
}

fn additive_expr(i: &str) -> ParseResult<Expr> {
    let (i, x) = multiplicative_expr(i)?;

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
            multiplicative_expr,
        ),
        x,
        move |xs, (op, y)| Expr::new(ExprKind::Binary(op, Box::new(xs), Box::new(y))),
    )(i)
}

fn expr(i: &str) -> ParseResult<Expr> {
    additive_expr(i)
}

fn equality(i: &str) -> ParseResult<Rel> {
    map(
        tuple((
            expr,
            delimited(
                space0,
                alt((
                    value(EqualityOp::Eq, tag("==")),
                    value(EqualityOp::Ge, tag(">=")),
                    value(EqualityOp::Gt, char('>')),
                    value(EqualityOp::Le, tag("<=")),
                    value(EqualityOp::Lt, char('<')),
                )),
                space0,
            ),
            expr,
        )),
        move |(x, op, y)| Rel::new(RelKind::Equality(op, Box::new(x), Box::new(y))),
    )(i)
}

fn primary_rel(i: &str) -> ParseResult<Rel> {
    alt((
        delimited(
            terminated(char('('), space0),
            rel,
            preceded(space0, char(')')),
        ),
        equality,
    ))(i)
}

fn and_rel(i: &str) -> ParseResult<Rel> {
    let (i, x) = primary_rel(i)?;

    fold_many0(
        preceded(delimited(space0, tag("&&"), space0), primary_rel),
        x,
        move |xs, y| Rel::new(RelKind::Binary(RelBinaryOp::And, Box::new(xs), Box::new(y))),
    )(i)
}

fn or_rel(i: &str) -> ParseResult<Rel> {
    let (i, x) = and_rel(i)?;

    fold_many0(
        preceded(delimited(space0, tag("||"), space0), and_rel),
        x,
        move |xs, y| Rel::new(RelKind::Binary(RelBinaryOp::Or, Box::new(xs), Box::new(y))),
    )(i)
}

fn rel(i: &str) -> ParseResult<Rel> {
    or_rel(i)
}

pub fn parse(i: &str) -> Result<Rel, String> {
    match all_consuming(rel)(i) {
        Ok(("", x)) => Ok(x),
        Err(NomErr::Error(e)) | Err(NomErr::Failure(e)) => Err(convert_error(i, e)),
        _ => unreachable!(),
    }
}

// Copied from `nom::error::convert_error`.
#[allow(clippy::naive_bytecount)]
fn convert_error(input: &str, e: VerboseError<&str>) -> String {
    use nom::Offset;

    let substring = e.errors.first().unwrap().0;
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
