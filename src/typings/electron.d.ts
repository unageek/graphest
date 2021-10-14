declare module "electron" {
  interface IpcMain {
    handle<T extends import("../common/ipc").MessageToMain>(
      channel: T["channel"],
      listener: (
        event: import("electron").IpcMainInvokeEvent,
        ...args: T["args"]
      ) => Promise<T["result"]>
    ): void;
  }

  interface WebContents {
    send<T extends import("../common/ipc").MessageToRenderer>(
      channel: T["channel"],
      ...args: T["args"]
    ): void;
  }
}
