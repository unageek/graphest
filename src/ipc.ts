import { IpcRendererEvent } from "electron";

export interface MessageToMain {
  channel: string;
  args: unknown[];
  result: unknown;
}

export const abortGraphing = "abort-graphing";
export interface AbortGraphing extends MessageToMain {
  channel: typeof abortGraphing;
  args: [relId: string, tileId?: string];
  result: void;
}

export const abortGraphingAll = "abort-graphing-all";
export interface AbortGraphingAll extends MessageToMain {
  channel: typeof abortGraphingAll;
  args: [];
  result: void;
}

export const newRelation = "new-relation";
export interface NewRelation extends MessageToMain {
  channel: typeof newRelation;
  args: [rel: string];
  result: { relId: string };
}

export const openUrl = "open-url";
export interface OpenUrl extends MessageToMain {
  channel: typeof openUrl;
  args: [url: string];
  result: void;
}

export const requestTile = "request-tile";
export interface RequestTile extends MessageToMain {
  channel: typeof requestTile;
  args: [relId: string, tileId: string, coords: L.Coords];
  result: void;
}

export const validateRelation = "validate-relation";
export interface ValidateRelation extends MessageToMain {
  channel: typeof validateRelation;
  args: [rel: string];
  result: { error?: string };
}

export interface MessageToRenderer {
  channel: string;
  args: unknown[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  listener: (event: IpcRendererEvent, ...args: any[]) => void;
}

export const graphingStatusChanged = "graphing-status-changed";
export interface GraphingStatusChanged extends MessageToRenderer {
  channel: typeof graphingStatusChanged;
  args: [relId: string, processing: boolean];
  listener: (
    event: IpcRendererEvent,
    ...args: GraphingStatusChanged["args"]
  ) => void;
}

export const tileReady = "tile-ready";
export interface TileReady extends MessageToRenderer {
  channel: typeof tileReady;
  args: [relId: string, tileId: string, url: string];
  listener: (event: IpcRendererEvent, ...args: TileReady["args"]) => void;
}

export interface IpcRenderer {
  invoke<T extends MessageToMain>(
    channel: T["channel"],
    ...args: T["args"]
  ): Promise<T["result"]>;

  on<T extends MessageToRenderer>(
    channel: T["channel"],
    listener: T["listener"]
  ): void;

  off<T extends MessageToRenderer>(
    channel: T["channel"],
    listener: T["listener"]
  ): void;
}
