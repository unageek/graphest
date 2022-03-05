/**
 * The zoom level where 1px corresponds to 1 unit length in real coordinates.
 *
 * In general, `2 ** (zoom - BASE_ZOOM_LEVEL)` pixels = 1 unit length,
 * or 1px = `2 ** (BASE_ZOOM_LEVEL - zoom)` unit lengths.
 *
 * As Leaflet does not handle negative zoom levels, we need some positive offset.
 */
export const BASE_ZOOM_LEVEL = 512;

/**
 * The zoom level in which the graph is initially shown.
 */
export const INITIAL_ZOOM_LEVEL = BASE_ZOOM_LEVEL + 6;

/**
 * The width/height of graph tiles in pixels.
 */
export const GRAPH_TILE_SIZE = 256;

/**
 * The extra width/height added to {@link GRAPH_TILE_SIZE} in pixels.
 *
 * The value is 1 and fixed, but defined as a constant for readability.
 *
 * Graph tiles are extended by 1px on the right and the bottom sides
 * then translated by -0.5px both horizontally and vertically
 * to place the origin at the corners of the center tiles.
 */
export const GRAPH_TILE_EXTENSION = 1;

/**
 * The sum of {@link GRAPH_TILE_SIZE} and {@link GRAPH_TILE_EXTENSION}.
 */
export const EXTENDED_GRAPH_TILE_SIZE = GRAPH_TILE_SIZE + GRAPH_TILE_EXTENSION;

/**
 * The default graph color.
 */
export const DEFAULT_COLOR = "rgba(0, 78, 140, 0.8)"; // `SharedColors.cyanBlue20`

/**
 * The maximum value allowed for a pen size.
 */
export const MAX_PEN_SIZE = 1000;

/**
 * The amount of coordinate perturbation in horizontal direction in pixels.
 */
export const PERTURBATION_X = 1.2345678901234567e-3;

/**
 * The amount of coordinate perturbation in vertical direction in pixels.
 */
export const PERTURBATION_Y = 1.3456789012345678e-3;
