export interface ExportImageEntry {
  color: string;
  relId: string;
}

export interface ExportImageOptions {
  antiAliasing: number;
  height: number;
  path: string;
  timeout: number;
  width: number;
  xMax: string;
  xMin: string;
  yMax: string;
  yMin: string;
}

export interface ExportImageProgress {
  lastStderr: string;
  lastUrl: string;
  progress: number;
}
