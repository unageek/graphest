import { IpcRendererEvent } from "electron";
import { Command } from "./command";
import {
  ExportImageEntry,
  ExportImageOptions,
  ExportImageProgress,
} from "./exportImage";
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

export const abortExportImage = "abort-export-image";
export interface AbortExportImage extends MessageToMain {
  channel: typeof abortExportImage;
  args: [];
  result: void;
}

export const abortGraphing = "abort-graphing";
export interface AbortGraphing extends MessageToMain {
  channel: typeof abortGraphing;
  args: [relId: string, tileId?: string];
  result: void;
}

export const exportImage = "export-image";
export interface ExportImage extends MessageToMain {
  channel: typeof exportImage;
  args: [entries: ExportImageEntry[], opts: ExportImageOptions];
  result: Promise<void>;
}

export const getDefaultImageFilePath = "get-pictures-folder";
export interface GetDefaultImageFilePath extends MessageToMain {
  channel: typeof getDefaultImageFilePath;
  args: [];
  result: Promise<string>;
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

export const showSaveDialog = "open-save-dialog";
export interface ShowSaveDialog extends MessageToMain {
  channel: typeof showSaveDialog;
  args: [path: string];
  result: Promise<string | undefined>;
}

export interface MessageToRenderer {
  channel: string;
  args: unknown[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  listener: (event: IpcRendererEvent, ...args: any[]) => void;
}

export const commandInvoked = "command-invoked";
export interface CommandInvoked extends MessageToRenderer {
  channel: typeof commandInvoked;
  args: [command: Command];
  listener: (event: IpcRendererEvent, ...args: CommandInvoked["args"]) => void;
}

export const exportImageStatusChanged = "export-image-status-changed";
export interface ExportImageStatusChanged extends MessageToRenderer {
  channel: typeof exportImageStatusChanged;
  args: [progress: ExportImageProgress];
  listener: (
    event: IpcRendererEvent,
    ...args: ExportImageStatusChanged["args"]
  ) => void;
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
