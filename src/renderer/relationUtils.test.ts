import { Range } from "../common/range";
import { Token, tokenize, TokenKind } from "./relationUtils";

test("tokenize", () => {
  const tokens = [...tokenize("y = sin(x)")];
  const expected: Token[] = [
    new Token(TokenKind.Identifier, new Range(0, 1), "y"),
    new Token(TokenKind.Space, new Range(1, 2), " "),
    new Token(TokenKind.Equals, new Range(2, 3), "="),
    new Token(TokenKind.Space, new Range(3, 4), " "),
    new Token(TokenKind.Identifier, new Range(4, 7), "sin"),
    new Token(TokenKind.LParen, new Range(7, 8), "("),
    new Token(TokenKind.Identifier, new Range(8, 9), "x"),
    new Token(TokenKind.RParen, new Range(9, 10), ")"),
  ];

  expect(tokens).toStrictEqual(expected);
});
