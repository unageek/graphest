import { contextBridge, ipcRenderer } from "electron";
import * as ipc from "../common/ipc";

const safeIpcRenderer: ipc.IpcRenderer = {
  invoke<T extends ipc.MessageToMain>(
    channel: T["channel"],
    ...args: T["args"]
  ): Promise<T["result"]> {
    return ipcRenderer.invoke(channel, ...args);
  },

  on<T extends ipc.MessageToRenderer>(
    channel: T["channel"],
    listener: T["listener"]
  ) {
    ipcRenderer.on(channel, listener);
  },

  off<T extends ipc.MessageToRenderer>(
    channel: T["channel"],
    listener: T["listener"]
  ) {
    ipcRenderer.off(channel, listener);
  },
};

contextBridge.exposeInMainWorld("ipcRenderer", safeIpcRenderer);
