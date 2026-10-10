import { MenuItemData } from "../common/ipc";
import {
  formatAccelerator,
  getAccessKey,
  getVisibleItemIndices,
  parseLabel,
} from "./menuUtils";

test("formatAccelerator", () => {
  expect(formatAccelerator("CmdOrCtrl+Shift+S")).toBe("Ctrl+Shift+S");
  expect(formatAccelerator("CommandOrControl+Z")).toBe("Ctrl+Z");
  expect(formatAccelerator("Shift+CommandOrControl+Z")).toBe("Ctrl+Shift+Z");
  expect(formatAccelerator("CmdOrCtrl+.")).toBe("Ctrl+.");
  expect(formatAccelerator("Alt+1")).toBe("Alt+1");
  expect(formatAccelerator("F11")).toBe("F11");
});

test("getAccessKey", () => {
  expect(getAccessKey("&File")).toBe("f");
  expect(getAccessKey("Save &As…")).toBe("a");
  expect(getAccessKey("Undo")).toBeUndefined();
});

test("getVisibleItemIndices", () => {
  const item = (type: MenuItemData["type"], visible = true): MenuItemData => ({
    checked: false,
    enabled: true,
    label: "",
    type,
    visible,
  });
  expect(
    getVisibleItemIndices([
      item("separator"),
      item("normal"),
      item("separator"),
      item("normal", false),
      item("separator"),
      item("normal"),
      item("separator"),
    ]),
  ).toStrictEqual([1, 2, 5]);
});

test("parseLabel", () => {
  expect(parseLabel("&File")).toStrictEqual({
    accessKeyIndex: 0,
    text: "File",
  });
  expect(parseLabel("A&bort Graphing")).toStrictEqual({
    accessKeyIndex: 1,
    text: "Abort Graphing",
  });
  expect(parseLabel("Undo")).toStrictEqual({ text: "Undo" });
});
