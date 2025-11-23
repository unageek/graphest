/**
 * Valid values for {@link ExportImageOptions.antiAliasing}.
 */
export const ANTI_ALIASING_OPTIONS = [1, 5, 9, 13, 17, 21, 25];

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
export const MAX_EXPORT_TIMEOUT = 300;

export interface ExportImageData {
  /** The background color of the image as a case-insensitive hex code (`#RRGGBBAA`). */
  background: string;
  /** The graphs in back-to-front order. */
  graphs: ExportImageGraph[];
}

export interface ExportImageGraph {
  /** The color of the graph as a case-insensitive hex code (`#RRGGBBAA`). */
  color: string;
  /** The pen size of the graph in pixels. */
  penSize: number;
  /** The ID of the relation. */
  relId: string;
}

export interface ExportImageOptions {
  /** The level of anti-aliasing. */
  antiAliasing: number;
  /** Perform alpha composition in linear color space. */
  correctAlpha: boolean;
  /** The height of the image in pixels. */
  height: number;
  /** The path where the image will be saved. */
  path: string;
  /** The per-tile timeout in seconds. */
  timeout: number;
  /** Make the image background transparent. */
  transparent: boolean;
  /** The width of the image in pixels. */
  width: number;
  /** The right bound of the graph. */
  xMax: string;
  /** The left bound of the graph. */
  xMin: string;
  /** The top bound of the graph. */
  yMax: string;
  /** The bottom bound of the graph. */
  yMin: string;
}

export interface ExportImageProgress {
  /** The standard error outputs generated during export. */
  messages: string[];
  /** The number of tiles to be rendered. */
  numTiles: number;
  /** The number of tiles rendered so far. */
  numTilesRendered: number;
}
