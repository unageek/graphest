import { Menu, MenuItemConstructorOptions, screen, shell } from "electron";
import { Command } from "../common/command";

export type MenuActions = {
  [key in Command]?: () => void;
};

export function createMainMenu(actions: MenuActions): Menu {
  // https://www.electronjs.org/docs/api/menu#examples
  // https://github.com/electron/electron/blob/main/lib/browser/api/menu-item-roles.ts
  const isMac = process.platform === "darwin";
  const isRetina = screen.getPrimaryDisplay().scaleFactor === 2;
  return Menu.buildFromTemplate([
    ...(isMac ? [{ role: "appMenu" }] : []),
    {
      role: "fileMenu",
      label: "&File",
      submenu: [
        {
          id: Command.NewDocument,
          label: "&New",
          accelerator: "CmdOrCtrl+N",
          click: actions[Command.NewDocument],
        },
        {
          id: Command.Open,
          label: "&Open…",
          accelerator: "CmdOrCtrl+O",
          click: actions[Command.Open],
        },
        // {
        //   id: Command.OpenFromClipboard,
        //   label: "Open from Clipboard",
        //   click: actions[Command.OpenFromClipboard],
        // },
        { type: "separator" },
        // The Close menu is required for closing the about panel.
        { role: "close" },
        {
          id: Command.Save,
          label: "&Save",
          accelerator: "CmdOrCtrl+S",
          click: actions[Command.Save],
        },
        {
          id: Command.SaveAs,
          label: "Save &As…",
          accelerator: "CmdOrCtrl+Shift+S",
          click: actions[Command.SaveAs],
        },
        // {
        //   id: Command.SaveToClipboard,
        //   label: "Save to Clipboard",
        //   click: actions[Command.SaveToClipboard],
        // },
        { type: "separator" },
        {
          id: Command.ExportImage,
          label: "Export as Image…",
          click: actions[Command.ExportImage],
        },
        { type: "separator" },
        ...(isMac ? [] : [{ role: "quit" }]),
      ],
    },
    { role: "editMenu", label: "&Edit" },
    {
      label: "&Graph",
      submenu: [
        {
          id: Command.ShowAxes,
          label: "Show &Axes",
          accelerator: isMac ? "Cmd+1" : "Alt+1",
          type: "checkbox",
          checked: true,
          click: actions[Command.ShowAxes],
        },
        {
          id: Command.ShowMajorGrid,
          label: "Show Major &Grid",
          accelerator: isMac ? "Cmd+2" : "Alt+2",
          type: "checkbox",
          checked: true,
          click: actions[Command.ShowMajorGrid],
        },
        {
          id: Command.ShowMinorGrid,
          label: "Show &Minor Grid",
          accelerator: isMac ? "Cmd+3" : "Alt+3",
          type: "checkbox",
          checked: true,
          click: actions[Command.ShowMinorGrid],
        },
        {
          type: "separator",
        },
        ...(isRetina
          ? [
              {
                id: Command.HighResolution,
                label: "&High Resolution",
                type: "checkbox",
                click: actions[Command.HighResolution],
              },
            ]
          : []),
        {
          type: "separator",
        },
        {
          id: Command.AbortGraphing,
          label: "A&bort Graphing",
          accelerator: "Esc",
          click: actions[Command.AbortGraphing],
        },
      ],
    },
    {
      role: "windowMenu",
      label: "&Window",
      submenu: [
        ...(isMac ? [{ role: "minimize" }, { role: "zoom" }] : []),
        // On macOS, it seems common to place the Toggle Full Screen menu under the Window menu
        // if there is nothing else to be placed under the View menu.
        { role: "togglefullscreen", label: "Toggle &Full Screen" },
        { type: "separator" },
        ...(isMac ? [{ role: "front" }] : []),
      ],
    },
    {
      role: "help",
      label: "&Help",
      submenu: [
        {
          label: "Graphest &Help",
          accelerator: isMac ? "" : "F1",
          click: async () => {
            await shell.openExternal(
              "https://unageek.github.io/graphest/guide/"
            );
          },
        },
        { type: "separator" },
        {
          label: "What's New",
          click: async () => {
            await shell.openExternal(
              "https://github.com/unageek/graphest/releases"
            );
          },
        },
        {
          label: "Example Relations",
          click: async () => {
            await shell.openExternal(
              "https://github.com/unageek/graphest/blob/main/Examples.md"
            );
          },
        },
        { type: "separator" },
        { role: "toggleDevTools" },
        { type: "separator" },
        ...(isMac ? [] : [{ role: "about" }]),
      ],
    },
  ] as MenuItemConstructorOptions[]);
}
