import * as ipc from "../common/ipc";
import { Range } from "../common/range";
import { ValidationResult } from "../common/validationResult";

const leftBracketKind = new Map([
  [")", "("],
  ["]", "["],
  ["⌉", "⌈"],
  ["⌋", "⌊"],
]);

export function getHighlights(
  rel: string,
  selection: Range
): { errors: Range[]; highlightsLeft: Range[]; highlightsRight: Range[] } {
  const errors: Range[] = [];
  const highlightsLeft: Range[] = [];
  const highlightsRight: Range[] = [];
  const leftBrackets: { kind: string; pos: number }[] = [];

  for (let pos = 0; pos < rel.length; pos++) {
    const kind = rel[pos];
    switch (kind) {
      case "(":
      case "[":
      case "⌈":
      case "⌊":
        leftBrackets.push({ kind, pos });
        break;
      case ")":
      case "]":
      case "⌉":
      case "⌋": {
        const left = leftBrackets.pop();
        if (!left || left.kind !== leftBracketKind.get(kind)) {
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

export function normalizeRelation(rel: string): string {
  return rel.replaceAll("-", "−"); // a hyphen-minus → a minus sign
}

export async function validateRelation(rel: string): Promise<ValidationResult> {
  return await window.ipcRenderer.invoke<ipc.ValidateRelation>(
    ipc.validateRelation,
    rel
  );
}
