import { IpcRendererEvent } from "electron";
import { Command } from "./command";
import { Document } from "./document";
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

export enum SaveTo {
  Clipboard = "clipboard",
  CurrentFile = "current-file",
  NewFile = "new-file",
}

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
  result: void;
}

export const getDefaultExportImagePath = "get-default-export-image-path";
export interface GetDefaultExportImagePath extends MessageToMain {
  channel: typeof getDefaultExportImagePath;
  args: [];
  result: string;
}

export const openUrl = "open-url";
export interface OpenUrl extends MessageToMain {
  channel: typeof openUrl;
  args: [url: string];
  result: void;
}

export const ready = "ready";
export interface Ready extends MessageToMain {
  channel: typeof ready;
  args: [];
  result: void;
}

export const requestRelation = "request-relation";
export interface RequestRelation extends MessageToMain {
  channel: typeof requestRelation;
  args: [rel: string, graphId: string, highRes: boolean];
  result: RequestRelationResult;
}

export const requestSave = "request-save";
export interface RequestSave extends MessageToMain {
  channel: typeof requestSave;
  args: [doc: Document, to: SaveTo];
  result: void;
}

export const requestTile = "request-tile";
export interface RequestTile extends MessageToMain {
  channel: typeof requestTile;
  args: [relId: string, tileId: string, coords: L.Coords];
  result: void;
}

export const requestUnload = "request-unload";
export interface RequestUnload extends MessageToMain {
  channel: typeof requestUnload;
  args: [doc: Document];
  result: void;
}

export const showSaveDialog = "show-save-dialog";
export interface ShowSaveDialog extends MessageToMain {
  channel: typeof showSaveDialog;
  args: [path: string];
  result: string | undefined;
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

export const initiateSave = "initiate-save";
export interface InitiateSave extends MessageToRenderer {
  channel: typeof initiateSave;
  args: [to: SaveTo];
  listener: (event: IpcRendererEvent, ...args: InitiateSave["args"]) => void;
}

export const initiateUnload = "initiate-unload";
export interface InitiateUnload extends MessageToRenderer {
  channel: typeof initiateUnload;
  args: [];
  listener: (event: IpcRendererEvent, ...args: InitiateUnload["args"]) => void;
}

export const load = "load";
export interface Load extends MessageToRenderer {
  channel: typeof load;
  args: [doc: Document];
  listener: (event: IpcRendererEvent, ...args: Load["args"]) => void;
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
