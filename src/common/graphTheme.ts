/** A set of colors used in graph views. */
export interface GraphTheme {
  /** The background color of graph views. */
  background: string;
  /** The color of axes and ticks and tick labels on visible axes. */
  foreground: string;
  /** The color of ticks and tick labels on hidden axes. */
  secondary: string;
  /** The color of grids. */
  tertiary: string;
}

export const StubGraphTheme: GraphTheme = {
  background: "",
  foreground: "",
  secondary: "",
  tertiary: "",
};
