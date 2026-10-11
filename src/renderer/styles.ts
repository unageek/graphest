import {
  Theme,
  tokens,
  webDarkTheme,
  webLightTheme,
} from "@fluentui/react-components";
import Color from "color";
import { CSSProperties, useSyncExternalStore } from "react";

export const frostedGlass = {
  backdropFilter: "blur(30px)",
  background: `rgb(from ${tokens.colorNeutralBackground2} r g b / var(--frostedGlassOpacity))`,
};

// The variables for `frostedGlass` and Fluent's subtle buttons on it.
export const frostedGlassVariables = (theme: Theme) => {
  const background = theme.colorNeutralBackground2;
  const opacity = new Color(background).isDark() ? 0.8 : 0.9;
  return {
    "--frostedGlassOpacity": String(opacity),
    // Fluent's subtle buttons layer their state colors on the background.
    // These tints make the buttons look as if the state colors were translucent as well.
    "--colorSubtleBackgroundHover": tint(
      background,
      theme.colorSubtleBackgroundHover,
      opacity,
    ),
    "--colorSubtleBackgroundPressed": tint(
      background,
      theme.colorSubtleBackgroundPressed,
      opacity,
    ),
  } as CSSProperties;
};

// Returns the color that, layered on `background` with `opacity`, looks like `color` with `opacity`.
// It has the least alpha possible, so that the backdrop shows through as much as possible.
const tint = (background: string, color: string, opacity: number) => {
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

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

const subscribeToDarkQuery = (onChange: () => void) => {
  darkQuery.addEventListener("change", onChange);
  return () => darkQuery.removeEventListener("change", onChange);
};

export const useSystemTheme = (): Theme =>
  useSyncExternalStore(subscribeToDarkQuery, () => darkQuery.matches)
    ? webDarkTheme
    : webLightTheme;
