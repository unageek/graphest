import { createTheme } from "@fluentui/react";

export const LightTheme = createTheme();

// https://www.aka.ms/themedesigner
//
//      Primary color: #2899f5
//         Text color: #ffffff
//   Background color: #1b1a19
export const DarkTheme = createTheme({
  palette: {
    themePrimary: "#2899f5",
    themeLighterAlt: "#02060a",
    themeLighter: "#061827",
    themeLight: "#0c2e49",
    themeTertiary: "#185b93",
    themeSecondary: "#2286d7",
    themeDarkAlt: "#3ca2f6",
    themeDark: "#59b0f7",
    themeDarker: "#84c5f9",
    neutralLighterAlt: "#262523",
    neutralLighter: "#2f2d2c",
    neutralLight: "#3d3b39",
    neutralQuaternaryAlt: "#464442",
    neutralQuaternary: "#4d4b49",
    neutralTertiaryAlt: "#6b6966",
    neutralTertiary: "#c8c8c8",
    neutralSecondary: "#d0d0d0",
    neutralSecondaryAlt: "#d0d0d0",
    neutralPrimaryAlt: "#dadada",
    neutralPrimary: "#ffffff",
    neutralDark: "#f4f4f4",
    black: "#f8f8f8",
    white: "#1b1a19",
  },
  isInverted: true,
});
