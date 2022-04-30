import { contextBridge, ipcRenderer } from "electron";
import * as ipc from "../common/ipc";
import { MessageToRenderer } from "../common/ipc";

const safeIpcRenderer: ipc.IpcRenderer = {
  invoke<T extends ipc.MessageToMain>(
    channel: T["channel"],
    ...args: T["args"]
  ): Promise<T["result"]> {
    return ipcRenderer.invoke(channel, ...args);
  },

  on<T extends MessageToRenderer>(
    channel: T["channel"],
    listener: ipc.RendererListener<T>
  ) {
    ipcRenderer.on(channel, listener);
  },

  off<T extends MessageToRenderer>(
    channel: T["channel"],
    listener: ipc.RendererListener<T>
  ) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ipcRenderer.off(channel, listener as (...args: any[]) => void);
  },
};

contextBridge.exposeInMainWorld("ipcRenderer", safeIpcRenderer);
