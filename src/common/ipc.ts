import { IpcRendererEvent } from "electron";
import { MenuItem } from "./MenuItem";
import { Range } from "./range";
import { Result } from "./result";

export interface RelationError {
  range: Range;
  message: string;
}

export type RequestRelationResult = Result<string, RelationError>;

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

export const openUrl = "open-url";
export interface OpenUrl extends MessageToMain {
  channel: typeof openUrl;
  args: [url: string];
  result: void;
}

export const requestRelation = "request-relation";
export interface RequestRelation extends MessageToMain {
  channel: typeof requestRelation;
  args: [rel: string, graphId: string, highRes: boolean];
  result: RequestRelationResult;
}

export const requestTile = "request-tile";
export interface RequestTile extends MessageToMain {
  channel: typeof requestTile;
  args: [relId: string, tileId: string, coords: L.Coords];
  result: void;
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

export const menuItemInvoked = "menu-item-invoked";
export interface MenuItemInvoked extends MessageToRenderer {
  channel: typeof menuItemInvoked;
  args: [item: MenuItem];
  listener: (event: IpcRendererEvent, ...args: MenuItemInvoked["args"]) => void;
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
