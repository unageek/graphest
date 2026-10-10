import { MenuItemData } from "../common/ipc";

const keyNames: Record<string, string> = {
  alt: "Alt",
  cmdorctrl: "Ctrl",
  commandorcontrol: "Ctrl",
  control: "Ctrl",
  ctrl: "Ctrl",
  meta: "Super",
  option: "Alt",
  plus: "+",
  shift: "Shift",
  super: "Super",
};

const modifierOrder = ["Ctrl", "Alt", "Shift", "Super"];

/**
 * Returns the accelerator in the form shown in menus on Windows and Linux, e.g., "Ctrl+Shift+S".
 */
export function formatAccelerator(accelerator: string): string {
  const keys = accelerator
    .split("+")
    .map((key) => keyNames[key.toLowerCase()] ?? key);
  const order = (key: string) => {
    const i = modifierOrder.indexOf(key);
    return i === -1 ? modifierOrder.length : i;
  };
  // The sort is stable, so the key stays last.
  return keys.sort((a, b) => order(a) - order(b)).join("+");
}

/**
 * Returns the access key of the label in lower case, e.g., "f" for "&File".
 */
export function getAccessKey(label: string): string | undefined {
  const { accessKeyIndex, text } = parseLabel(label);
  return accessKeyIndex !== undefined
    ? text[accessKeyIndex].toLowerCase()
    : undefined;
}

/**
 * Returns the indices of the items to show,
 * omitting hidden items and leading, trailing, and consecutive separators.
 */
export function getVisibleItemIndices(items: MenuItemData[]): number[] {
  const indices: number[] = [];
  const lastIsSeparator = () =>
    items[indices[indices.length - 1]]?.type === "separator";
  for (const [i, item] of items.entries()) {
    const redundant =
      item.type === "separator" && (indices.length === 0 || lastIsSeparator());
    if (item.visible && !redundant) {
      indices.push(i);
    }
  }
  if (lastIsSeparator()) {
    indices.pop();
  }
  return indices;
}

/**
 * Returns the label without the "&" that marks the access key, and the position of the access key.
 */
export function parseLabel(label: string): {
  accessKeyIndex?: number;
  text: string;
} {
  const i = label.indexOf("&");
  if (i === -1 || i === label.length - 1) {
    return { text: label };
  }
  return { accessKeyIndex: i, text: label.slice(0, i) + label.slice(i + 1) };
}
