import * as ipc from "../common/ipc";

export function normalizeRelation(rel: string): string {
  return rel.replaceAll("-", "−"); // a hyphen-minus → a minus sign
}

const openOp = new Map([
  [")", "("],
  ["]", "["],
  ["⌉", "⌈"],
  ["⌋", "⌊"],
]);

export function syntaxHighlight(
  rel: string,
  selection: [number, number]
): { error: number[]; highlight: number[] } {
  const openOps: { op: string; pos: number }[] = [];
  const errors: number[] = [];
  const highlights: number[] = [];

  for (let i = 0; i < rel.length; i++) {
    const c = rel[i];
    switch (c) {
      case "(":
      case "[":
      case "⌈":
      case "⌊":
        openOps.push({ op: c, pos: i });
        break;
      case ")":
      case "]":
      case "⌉":
      case "⌋": {
        const open = openOps.pop();
        if (!open || open.op !== openOp.get(c)) {
          errors.splice(-1, 0, ...openOps.map((o) => o.pos));
          if (open) errors.push(open.pos);
          errors.push(i);
          openOps.length = 0;
        } else if (highlights.length === 0) {
          if (open.pos < selection[0] && selection[1] <= i) {
            highlights.push(open.pos);
            highlights.push(i);
          }
        }
        break;
      }
    }
  }

  errors.splice(-1, 0, ...openOps.map((o) => o.pos));

  return { error: errors, highlight: highlights };
}

export async function validateRelation(rel: string): Promise<boolean> {
  const { error } = await window.ipcRenderer.invoke<ipc.ValidateRelation>(
    ipc.validateRelation,
    rel
  );
  return error === undefined;
}
