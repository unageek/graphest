import * as ipc from "../common/ipc";
import { Range } from "../common/range";
import { ValidationResult } from "../common/validationResult";

export enum TokenKind {
  /** `&&` or `∧`. */
  And,
  /** `*`. */
  Asterisk,
  /** `|`. */
  Bar,
  /** `^`. */
  Caret,
  /** `,`. */
  Comma,
  /** `=`. */
  Equals,
  /** `>`. */
  GreaterThan,
  /** `>=` or `≥`. */
  GreaterThanOrEquals,
  /** An identifier. */
  Identifier,
  /** `[`. */
  LBracket,
  /** `⌈`. */
  LCeil,
  /** `<`. */
  LessThan,
  /** `<=` or `≤`. */
  LessThanOrEquals,
  /** `⌊`. */
  LFloor,
  /** `(`. */
  LParen,
  /** `-` (U+002D) or `−` (U+2212). */
  Minus,
  /** `!` or `¬`. */
  Not,
  /** A number literal. */
  Number,
  /** `∨`. Note that `||` is treated as two {@link Bar}s. */
  Or,
  /** `+`. */
  Plus,
  /** `]`. */
  RBracket,
  /** `⌉`. */
  RCeil,
  /** `⌋`. */
  RFloor,
  /** `)`. */
  RParen,
  /** `/`. */
  Slash,
  /** A space or a horizontal tab. */
  Space,
  /** `~`. */
  Tilde,
  /** An unknown token. */
  Unknown,
}

export type Token = {
  kind: TokenKind;
  range: Range;
  source: string;
};

export const NormalizationRules: [string, string][] = [
  ["-", "−"], // a hyphen-minus → a minus sign
  ["<=", "≤"],
  [">=", "≥"],
  [" ", " "], // a non-breaking space → a space
  ["\t", "    "], // a horizontal tab → four spaces
  ["\r\n", " "],
  ["\r", " "],
  ["\n", " "],
];

const leftBracketToRight = new Map([
  ["(", ")"],
  ["[", "]"],
  ["⌈", "⌉"],
  ["⌊", "⌋"],
]);

const rightBracketToLeft = new Map([
  [")", "("],
  ["]", "["],
  ["⌉", "⌈"],
  ["⌋", "⌊"],
]);

export function areBracketsBalanced(tokens: Token[]): boolean {
  const leftBrackets: Token[] = [];

  for (const token of tokens) {
    switch (token.kind) {
      case TokenKind.LBracket:
      case TokenKind.LCeil:
      case TokenKind.LFloor:
      case TokenKind.LParen:
        leftBrackets.push(token);
        break;

      case TokenKind.RBracket:
      case TokenKind.RCeil:
      case TokenKind.RFloor:
      case TokenKind.RParen: {
        const left = leftBrackets.pop();
        if (!left || left.source !== getLeftBracket(token.source)) {
          return false;
        }
        break;
      }
    }
  }

  return leftBrackets.length === 0;
}

const EXPECTED_TOKENS_AFTER_LEFT_BRACKET = new Set([
  TokenKind.Bar,
  TokenKind.Identifier,
  TokenKind.LBracket,
  TokenKind.LCeil,
  TokenKind.LFloor,
  TokenKind.LParen,
  TokenKind.Minus,
  TokenKind.Not,
  TokenKind.Number,
  TokenKind.Plus,
  TokenKind.RBracket,
  TokenKind.RCeil,
  TokenKind.RFloor,
  TokenKind.RParen,
  TokenKind.Space,
  TokenKind.Tilde,
]);

const EXPECTED_TOKENS_BEFORE_RIGHT_BRACKETS = new Set([
  TokenKind.Bar,
  TokenKind.Identifier,
  TokenKind.LBracket,
  TokenKind.LCeil,
  TokenKind.LFloor,
  TokenKind.LParen,
  TokenKind.Number,
  TokenKind.RBracket,
  TokenKind.RCeil,
  TokenKind.RFloor,
  TokenKind.RParen,
  TokenKind.Space,
]);

export function getDecorations(
  tokens: Token[],
  selection: Range
): {
  highlightedLeftBrackets: Token[];
  highlightedRightBrackets: Token[];
  multiplications: Token[];
  syntaxErrors: Token[];
} {
  const highlightedLeftBrackets: Token[] = [];
  const highlightedRightBrackets: Token[] = [];
  const multiplications: Token[] = [];
  const syntaxErrors: Token[] = [];

  const leftBrackets: Token[] = [];

  {
    const next = tokens[0];
    if (next && !EXPECTED_TOKENS_AFTER_LEFT_BRACKET.has(next.kind)) {
      syntaxErrors.push(next);
    }
  }

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const prev = tokens[i - 1];
    const next = tokens[i + 1];

    switch (token.kind) {
      case TokenKind.LBracket:
      case TokenKind.LCeil:
      case TokenKind.LFloor:
      case TokenKind.LParen: {
        leftBrackets.push(token);

        if (next && !EXPECTED_TOKENS_AFTER_LEFT_BRACKET.has(next.kind)) {
          syntaxErrors.push(next);
        }
        break;
      }

      case TokenKind.RBracket:
      case TokenKind.RCeil:
      case TokenKind.RFloor:
      case TokenKind.RParen: {
        const left = leftBrackets.pop();
        if (!left || left.source !== getLeftBracket(token.source)) {
          syntaxErrors.push(...leftBrackets);
          if (left) {
            syntaxErrors.push(left);
          }
          syntaxErrors.push(token);
          leftBrackets.length = 0;
        } else if (highlightedLeftBrackets.length === 0) {
          if (
            left.range.start < selection.start &&
            selection.end <= token.range.start
          ) {
            highlightedLeftBrackets.push(left);
            highlightedRightBrackets.push(token);
          }
        }

        if (prev && !EXPECTED_TOKENS_BEFORE_RIGHT_BRACKETS.has(prev.kind)) {
          syntaxErrors.push(prev);
        }
        break;
      }

      case TokenKind.Space: {
        if (
          prev?.kind === TokenKind.Number &&
          next?.kind === TokenKind.Number
        ) {
          multiplications.push(token);
        }
        break;
      }

      case TokenKind.Unknown:
        syntaxErrors.push(token);
        break;
    }
  }

  syntaxErrors.push(...leftBrackets);

  {
    const prev = tokens[tokens.length - 1];
    if (prev && !EXPECTED_TOKENS_BEFORE_RIGHT_BRACKETS.has(prev.kind)) {
      syntaxErrors.push(prev);
    }
  }

  return {
    highlightedLeftBrackets,
    highlightedRightBrackets,
    multiplications,
    syntaxErrors,
  };
}

export function getLeftBracket(rightBracket: string): string | undefined {
  return rightBracketToLeft.get(rightBracket);
}

export function getRightBracket(leftBracket: string): string | undefined {
  return leftBracketToRight.get(leftBracket);
}

export function isLeftBracket(ch: string): boolean {
  return leftBracketToRight.has(ch);
}

export function isRightBracket(ch: string): boolean {
  return rightBracketToLeft.has(ch);
}

const IDENTIFIER = /^\p{Alphabetic}[\p{Alphabetic}\d']*/u;
const NUMBER_LITERAL = /^(?:\d+\.?\d*|\.\d+)/;

export function* tokenize(rel: string): Generator<Token, void> {
  let i = 0;
  while (i < rel.length) {
    let kind: TokenKind | undefined = undefined;

    switch (rel[i]) {
      case " ":
      case "\t":
        kind = TokenKind.Space;
        break;
      case "∧":
        kind = TokenKind.And;
        break;
      case "*":
        kind = TokenKind.Asterisk;
        break;
      case "|":
        kind = TokenKind.Bar;
        break;
      case "^":
        kind = TokenKind.Caret;
        break;
      case ",":
        kind = TokenKind.Comma;
        break;
      case "=":
        kind = TokenKind.Equals;
        break;
      case ">":
        kind = TokenKind.GreaterThan;
        break;
      case "≥":
        kind = TokenKind.GreaterThanOrEquals;
        break;
      case "[":
        kind = TokenKind.LBracket;
        break;
      case "⌈":
        kind = TokenKind.LCeil;
        break;
      case "<":
        kind = TokenKind.LessThan;
        break;
      case "≤":
        kind = TokenKind.LessThanOrEquals;
        break;
      case "⌊":
        kind = TokenKind.LFloor;
        break;
      case "(":
        kind = TokenKind.LParen;
        break;
      case "-":
      case "−":
        kind = TokenKind.Minus;
        break;
      case "!":
      case "¬":
        kind = TokenKind.Not;
        break;
      case "∨":
        kind = TokenKind.Or;
        break;
      case "+":
        kind = TokenKind.Plus;
        break;
      case "]":
        kind = TokenKind.RBracket;
        break;
      case "⌉":
        kind = TokenKind.RCeil;
        break;
      case "⌋":
        kind = TokenKind.RFloor;
        break;
      case ")":
        kind = TokenKind.RParen;
        break;
      case "/":
        kind = TokenKind.Slash;
        break;
      case "~":
        kind = TokenKind.Tilde;
        break;
    }
    if (kind !== undefined) {
      yield { kind, range: new Range(i, i + 1), source: rel[i] };
      i++;
      continue;
    }

    switch (rel.slice(i, i + 2)) {
      case "&&":
        kind = TokenKind.And;
        break;
      case ">=":
        kind = TokenKind.GreaterThanOrEquals;
        break;
      case "<=":
        kind = TokenKind.LessThanOrEquals;
        break;
    }
    if (kind !== undefined) {
      yield { kind, range: new Range(i, i + 2), source: rel.slice(i, i + 2) };
      i += 2;
      continue;
    }

    const forward = rel.slice(i);

    const identifier = forward.match(IDENTIFIER);
    if (identifier) {
      const len = identifier[0].length;
      yield {
        kind: TokenKind.Identifier,
        range: new Range(i, i + len),
        source: identifier[0],
      };
      i += len;
      continue;
    }

    const numberLiteral = forward.match(NUMBER_LITERAL);
    if (numberLiteral) {
      const len = numberLiteral[0].length;
      yield {
        kind: TokenKind.Number,
        range: new Range(i, i + len),
        source: numberLiteral[0],
      };
      i += len;
      continue;
    }

    yield {
      kind: TokenKind.Unknown,
      range: new Range(i, i + 1),
      source: rel.slice(i, i + 1),
    };
    i++;
  }
}

export async function validateRelation(rel: string): Promise<ValidationResult> {
  return await window.ipcRenderer.invoke<ipc.ValidateRelation>(
    ipc.validateRelation,
    rel
  );
}
