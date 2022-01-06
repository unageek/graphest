import * as ipc from "../common/ipc";
import { Range } from "../common/range";
import { ValidationResult } from "../common/validationResult";

export const NormalizationRules: [string, string][] = [
  ["-", "−"], // hyphen-minus → minus sign
  ["<=", "≤"],
  [">=", "≥"],
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

export function areBracketsBalanced(rel: string): boolean {
  const leftBrackets: { ch: string; pos: number }[] = [];

  for (let pos = 0; pos < rel.length; pos++) {
    const ch = rel[pos];
    switch (ch) {
      case "(":
      case "[":
      case "⌈":
      case "⌊":
        leftBrackets.push({ ch, pos });
        break;
      case ")":
      case "]":
      case "⌉":
      case "⌋": {
        const left = leftBrackets.pop();
        if (!left || left.ch !== getLeftBracket(ch)) {
          return false;
        }
        break;
      }
    }
  }

  return leftBrackets.length === 0;
}

export function getLeftBracket(rightBracket: string): string | undefined {
  return rightBracketToLeft.get(rightBracket);
}

export function getRightBracket(leftBracket: string): string | undefined {
  return leftBracketToRight.get(leftBracket);
}

export function highlightBrackets(
  rel: string,
  selection: Range
): { errors: Range[]; highlightsLeft: Range[]; highlightsRight: Range[] } {
  const errors: Range[] = [];
  const highlightsLeft: Range[] = [];
  const highlightsRight: Range[] = [];
  const leftBrackets: { ch: string; pos: number }[] = [];

  for (let pos = 0; pos < rel.length; pos++) {
    const ch = rel[pos];
    switch (ch) {
      case "(":
      case "[":
      case "⌈":
      case "⌊":
        leftBrackets.push({ ch, pos });
        break;
      case ")":
      case "]":
      case "⌉":
      case "⌋": {
        const left = leftBrackets.pop();
        if (!left || left.ch !== getLeftBracket(ch)) {
          errors.push(...leftBrackets.map((l) => new Range(l.pos, l.pos + 1)));
          if (left) {
            errors.push(new Range(left.pos, left.pos + 1));
          }
          errors.push(new Range(pos, pos + 1));
          leftBrackets.length = 0;
        } else if (highlightsLeft.length === 0) {
          if (left.pos < selection.start && selection.end <= pos) {
            highlightsLeft.push(new Range(left.pos, left.pos + 1));
            highlightsRight.push(new Range(pos, pos + 1));
          }
        }
        break;
      }
    }
  }

  errors.push(...leftBrackets.map((l) => new Range(l.pos, l.pos + 1)));

  return { errors, highlightsLeft, highlightsRight };
}

export function isLeftBracket(ch: string): boolean {
  return leftBracketToRight.has(ch);
}

export function isRightBracket(ch: string): boolean {
  return rightBracketToLeft.has(ch);
}

export async function validateRelation(rel: string): Promise<ValidationResult> {
  return await window.ipcRenderer.invoke<ipc.ValidateRelation>(
    ipc.validateRelation,
    rel
  );
}
