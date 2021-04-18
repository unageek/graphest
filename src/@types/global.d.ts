import * as ipc from "../ipc";

declare module "electron" {
  interface IpcMain {
    handle<T extends ipc.MessageToMain>(
      channel: T["channel"],
      listener: (
        event: IpcMainInvokeEvent,
        ...args: T["args"]
      ) => Promise<T["result"]>
    ): void;
  }

  interface WebContents {
    send<T extends ipc.MessageToRenderer>(
      channel: T["channel"],
      ...args: T["args"]
    ): void;
  }
}

declare global {
  interface Window {
    ipcRenderer: ipc.IpcRenderer;
  }
}
