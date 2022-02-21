/**
 * Valid values for {@link ExportImageOptions.antiAliasing}.
 */
export const ANTI_ALIASING_OPTIONS = [1, 3, 5, 7, 9, 11, 13, 15, 17];

/**
 * The width/height of graph tiles used for exporting in pixels.
 */
export const EXPORT_GRAPH_TILE_SIZE = 1024;

/**
 * The maximum values for {@link ExportImageOptions.width} and {@link ExportImageOptions.height}.
 */
export const MAX_EXPORT_IMAGE_SIZE = 16384;

/**
 * The maximum value for {@link ExportImageOptions.timeout}.
 */
export const MAX_EXPORT_TIMEOUT = 300000;

export interface ExportImageEntry {
  color: string;
  relId: string;
  thickness: number;
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
