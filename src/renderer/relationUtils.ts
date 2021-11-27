import * as ipc from "../common/ipc";

export interface Range {
  start: number;
  end: number;
}

const leftBracketKind = new Map([
  [")", "("],
  ["]", "["],
  ["⌉", "⌈"],
  ["⌋", "⌊"],
]);

export function getHighlights(
  rel: string,
  selection: [number, number]
): { errors: Range[]; highlights: Range[] } {
  const errors: Range[] = [];
  const highlights: Range[] = [];
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
          errors.splice(
            -1,
            0,
            ...leftBrackets.map((l) => ({
              start: l.pos,
              end: l.pos + 1,
            }))
          );
          if (left) {
            errors.push({
              start: left.pos,
              end: left.pos + 1,
            });
          }
          errors.push({
            start: pos,
            end: pos + 1,
          });
          leftBrackets.length = 0;
        } else if (highlights.length === 0) {
          if (left.pos < selection[0] && selection[1] <= pos) {
            highlights.push({
              start: left.pos,
              end: left.pos + 1,
            });
            highlights.push({
              start: pos,
              end: pos + 1,
            });
          }
        }
        break;
      }
    }
  }

  errors.splice(
    -1,
    0,
    ...leftBrackets.map((l) => ({
      start: l.pos,
      end: l.pos + 1,
    }))
  );

  return { errors, highlights };
}

export function normalizeRelation(rel: string): string {
  return rel.replaceAll("-", "−"); // a hyphen-minus → a minus sign
}

export async function validateRelation(rel: string): Promise<boolean> {
  const { error } = await window.ipcRenderer.invoke<ipc.ValidateRelation>(
    ipc.validateRelation,
    rel
  );
  return error === undefined;
}
