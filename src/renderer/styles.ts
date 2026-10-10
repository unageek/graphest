import { Theme, tokens } from "@fluentui/react-components";
import Color from "color";
import { CSSProperties } from "react";

const opacity = 0.9;

export const translucentBackground = {
  backdropFilter: "blur(30px)",
  background: `rgb(from ${tokens.colorNeutralBackground2} r g b / ${opacity})`,
};

// Fluent's subtle buttons layer their state colors on the background. These tints make the buttons
// on `translucentBackground` look as if the state colors were translucent as well.
export const translucentSubtleStates = (theme: Theme) =>
  ({
    "--colorSubtleBackgroundHover": tint(
      theme.colorNeutralBackground2,
      theme.colorSubtleBackgroundHover,
    ),
    "--colorSubtleBackgroundPressed": tint(
      theme.colorNeutralBackground2,
      theme.colorSubtleBackgroundPressed,
    ),
  }) as CSSProperties;

// Returns the color that, layered on `background` with `opacity`, looks like `color` with `opacity`.
// It has the least alpha possible, so that the backdrop shows through as much as possible.
const tint = (background: string, color: string) => {
  const bg = new Color(background).unitArray();
  const fg = new Color(color).unitArray();
  const alpha = Math.max(
    ...bg.map((b, i) =>
      fg[i] < b ? 1 - fg[i] / b : (opacity * (fg[i] - b)) / (1 - opacity * b),
    ),
  );
  if (alpha === 0) {
    return "transparent";
  }
  return Color.rgb(
    bg.map((b, i) => (255 * opacity * (fg[i] - (1 - alpha) * b)) / alpha),
  )
    .alpha(alpha)
    .string();
};
