import {
  AlphaSlider,
  ColorArea,
  ColorPicker,
  ColorSlider,
  Input,
  Label,
  tokens,
} from "@fluentui/react-components";
import * as Color from "color";
import * as React from "react";

export interface MyColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const CustomColorPicker = (props: MyColorPickerProps): JSX.Element => {
  const color = new Color(props.color);

  return (
    <div
      style={{
        // Adjusted to the width of the `SwatchColorPicker`.
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        width: "284px",
      }}
    >
      <ColorPicker
        color={{
          h: color.hue(),
          s: 0.01 * color.saturationv(),
          v: 0.01 * color.value(),
          a: color.alpha(),
        }}
        onColorChange={(_, { color }) =>
          props.onColorChanged(
            Color.hsv({
              h: color.h,
              s: 100 * color.s,
              v: 100 * color.v,
              alpha: color.a,
            }).hexa()
          )
        }
      >
        <ColorArea
          style={{
            aspectRatio: "2 / 1",
            minHeight: "unset",
            minWidth: "unset",
            width: "100%",
          }}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            gap: "10px",
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              flexGrow: 1,
              gap: "10px",
            }}
          >
            <ColorSlider
              style={{
                minHeight: "unset",
              }}
              rail={{
                style: {
                  boxSizing: "border-box",
                  border: `1px solid ${tokens.colorNeutralStroke1}`,
                },
              }}
            />
            <AlphaSlider
              style={{
                minHeight: "unset",
              }}
              rail={{
                style: {
                  boxSizing: "border-box",
                  border: `1px solid ${tokens.colorNeutralStroke1}`,
                },
              }}
            />
          </div>
          <div
            style={{
              background: color.hex(),
              border: `1px solid ${tokens.colorNeutralStroke1}`,
              borderRadius: tokens.borderRadiusMedium,
              boxSizing: "border-box",
              height: "50px",
              width: "50px",
            }}
          />
        </div>
      </ColorPicker>
      <div
        style={{
          alignItems: "baseline",
          display: "flex",
          flexDirection: "row",
          gap: "8px",
        }}
      >
        <Label>Hex:</Label>
        <Input
          contentBefore="#"
          onChange={(_, { value }) => {
            const color = new Color(value);
            props.onColorChanged(color.hex());
          }}
          style={{ width: "100px" }}
          value={color.hex().substring(1)}
        />
      </div>
    </div>
  );
};
