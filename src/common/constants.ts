/**
 * The zoom level where the width/height of graph tiles correspond to unit length in real coordinates.
 */
export const BASE_ZOOM_LEVEL = 512;

/**
 * The zoom level in which the graph is initially shown.
 */
export const INITIAL_ZOOM_LEVEL = BASE_ZOOM_LEVEL - 2;

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