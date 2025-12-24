import {
  AlphaSlider,
  ColorArea,
  ColorPicker,
  ColorSlider,
  Input,
  tokens,
} from "@fluentui/react-components";
import Color from "color";
import { ReactNode, useState } from "react";

export interface CustomColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const CustomColorPicker = (props: CustomColorPickerProps): ReactNode => {
  const color = new Color(props.color);
  const [hex, setHex] = useState(color.hex().substring(1));
  const [hue, setHue] = useState(color.hue());
  const [saturation, setSaturation] = useState(0.01 * color.saturationv());
  const [value, setValue] = useState(0.01 * color.value());
  const [alpha, setAlpha] = useState(color.alpha());

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
          alignItems: "center",
          display: "grid",
          gap: "8px",
          gridTemplateColumns: "auto auto",
        }}
      >
        <Input
          contentBefore="#"
          onChange={(_, { value }) => {
            setHex(value);
            try {
              const c = new Color(`#${value}`);
              setHue(c.hue());
              setSaturation(0.01 * c.saturationv());
              setValue(0.01 * c.value());
              props.onColorChanged(c.hexa());
            } catch (_e) {
              // ignore invalid color
            }
          }}
          style={{ minWidth: 0 }}
          value={hex}
        />
        <Input
          contentAfter="%"
          onChange={(_, { value }) => {
            const a = Math.max(0, Math.min(1, Number(value) / 100));
            if (Number.isNaN(a) || a === alpha) {
              return;
            }
            const c = color.alpha(a);
            setAlpha(a);
            props.onColorChanged(c.hexa());
          }}
          style={{ minWidth: 0 }}
          value={Math.round(100 * alpha).toString()}
        />
      </div>
    </div>
  );
};
