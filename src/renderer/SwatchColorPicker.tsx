import {
  renderSwatchPickerGrid,
  SwatchPicker,
} from "@fluentui/react-components";
import Color from "color";
import * as React from "react";

export interface SwatchColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const SwatchColorPicker = (
  props: SwatchColorPickerProps,
): React.ReactNode => {
  const color = new Color(props.color);

  return (
    <SwatchPicker
      layout="grid"
      onSelectionChange={(_, { selectedSwatch }) => {
        if (selectedSwatch !== undefined) {
          const c = new Color(selectedSwatch).alpha(color.alpha());
          props.onColorChanged(c.hexa());
        }
      }}
      selectedValue={color.hex()}
      size="extra-small"
    >
      {renderSwatchPickerGrid({
        items: swatchColors.map((c) => ({
          color: c.hex(),
          value: c.hex(),
        })),
        columnCount: 12,
      })}
    </SwatchPicker>
  );
};

const swatchColors: Color[] = [
  "#750b1c",
  "#a4262c",
  "#d13438",
  "#603d30",
  "#da3b01",
  "#8e562e",
  "#ca5010",
  "#ffaa44",
  "#fce100",
  "#986f0b",
  "#c19c00",
  "#8cbd18",
  "#0b6a0b",
  "#498205",
  "#00ad56",
  "#005e50",
  "#005b70",
  "#038387",
  "#00b7c3",
  "#004e8c",
  "#0078d4",
  "#4f6bed",
  "#373277",
  "#5c2e91",
  "#8764b8",
  "#8378de",
  "#881798",
  "#c239b3",
  "#9b0062",
  "#e3008c",
  "#393939",
  "#7a7574",
  "#69797e",
  "#a0aeb2",
].map((c) => new Color(c));
