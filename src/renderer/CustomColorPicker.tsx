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

export interface CustomColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const CustomColorPicker = (
  props: CustomColorPickerProps
): JSX.Element => {
  const color = new Color(props.color);
  const [hex, setHex] = React.useState(color.hex().substring(1));
  const [hue, setHue] = React.useState(color.hue());
  const [saturation, setSaturation] = React.useState(
    0.01 * color.saturationv()
  );
  const [value, setValue] = React.useState(0.01 * color.value());
  const [alpha, setAlpha] = React.useState(color.alpha());

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
          h: hue,
          s: saturation,
          v: value,
          a: alpha,
        }}
        onColorChange={(_, { color }) => {
          setHue(color.h);
          setSaturation(color.s);
          setValue(color.v);
          setAlpha(Number(color.a));

          const c = Color.hsv({
            h: color.h,
            s: 100 * color.s,
            v: 100 * color.v,
            alpha: color.a,
          });
          setHex(c.hex().substring(1));
          props.onColorChanged(c.hexa());
        }}
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
              background: color.hexa(),
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
            setHex(value);
            try {
              const c = new Color(`#${value}`);
              setHue(c.hue());
              setSaturation(0.01 * c.saturationv());
              setValue(0.01 * c.value());
              props.onColorChanged(c.hex());
            } catch (e) {
              // ignore invalid color
            }
          }}
          style={{ width: "100px" }}
          value={hex}
        />
      </div>
    </div>
  );
};
