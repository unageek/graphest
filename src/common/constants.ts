// At zoom level 0, 1 pixel corresponds to 1 real coordinate length. The followings hold:
// 1 coordinate length = 2^(zoom level) pixels; or equivalently,
// 1 pixel = 2^-(zoom level) coordinate lengths.

/**
 * The z coordinate of Leaflet maps at zoom level 0 so that `z = zoomLevel + LEAFLET_Z_OFFSET`.
 *
 * As Leaflet does not handle negative z coordinates, we need some positive offset.
 */
export const LEAFLET_Z_OFFSET = 512;

/**
 * The zoom level in which the graph is initially shown.
 */
export const INITIAL_ZOOM_LEVEL = 6;

/**
 * The minimum zoom level.
 */
export const MIN_ZOOM_LEVEL = -LEAFLET_Z_OFFSET;

/**
 * The maximum zoom level.
 *
 * The z coordinate of Leaflet maps cannot exceed 1023.
 */
export const MAX_ZOOM_LEVEL = 1023 - LEAFLET_Z_OFFSET;

/**
 * The maximum absolute value of pixel coordinates of Leaflet maps.
 *
 * Up to this value, integers can be represented exactly.
 * Leaflet maps can get stuck if pixel coordinates exceed it.
 */
export const MAX_PIXEL_COORDINATE = 2 ** 53;

/**
 * Returns the maximum absolute value of coordinates that can be shown at the given zoom level.
 */
export function maxCoordinate(zoomLevel: number): number {
  return MAX_PIXEL_COORDINATE * 2 ** -zoomLevel;
}

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
export const DEFAULT_PEN_COLOR = "rgba(0, 78, 140, 0.8)"; // `SharedColors.cyanBlue20`

/**
 * The maximum value allowed for a pen thickness.
 */
export const MAX_PEN_THICKNESS = 1000;

/**
 * The amount of coordinate perturbation in horizontal direction in pixels.
 */
export const PERTURBATION_X = 1.2345678901234567e-3;

/**
 * The amount of coordinate perturbation in vertical direction in pixels.
 */
export const PERTURBATION_Y = 1.3456789012345678e-3;
