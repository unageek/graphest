import { Range } from "../common/range";
import { tokenize, TokenKind } from "./relationUtils";

test("tokenize", () => {
  const tokens = tokenize("y = sin(x)");
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Identifier,
    range: new Range(0, 1),
    slice: "y",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Space,
    range: new Range(1, 2),
    slice: " ",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Equals,
    range: new Range(2, 3),
    slice: "=",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Space,
    range: new Range(3, 4),
    slice: " ",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Identifier,
    range: new Range(4, 7),
    slice: "sin",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.LParen,
    range: new Range(7, 8),
    slice: "(",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.Identifier,
    range: new Range(8, 9),
    slice: "x",
  });
  expect(tokens.next().value).toStrictEqual({
    kind: TokenKind.RParen,
    range: new Range(9, 10),
    slice: ")",
  });
  expect(tokens.next().done).toBe(true);
});
