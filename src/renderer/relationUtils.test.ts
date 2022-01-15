import { Range } from "../common/range";
import { Token, tokenize, TokenKind } from "./relationUtils";

test("tokenize", () => {
  const tokens = [...tokenize("y = sin(x)")];
  const expected: Token[] = [
    {
      kind: TokenKind.Identifier,
      range: new Range(0, 1),
      source: "y",
    },
    {
      kind: TokenKind.Space,
      range: new Range(1, 2),
      source: " ",
    },
    {
      kind: TokenKind.Equals,
      range: new Range(2, 3),
      source: "=",
    },
    {
      kind: TokenKind.Space,
      range: new Range(3, 4),
      source: " ",
    },
    {
      kind: TokenKind.Identifier,
      range: new Range(4, 7),
      source: "sin",
    },
    {
      kind: TokenKind.LParen,
      range: new Range(7, 8),
      source: "(",
    },
    {
      kind: TokenKind.Identifier,
      range: new Range(8, 9),
      source: "x",
    },
    {
      kind: TokenKind.RParen,
      range: new Range(9, 10),
      source: ")",
    },
  ];

  expect(tokens).toStrictEqual(expected);
});
