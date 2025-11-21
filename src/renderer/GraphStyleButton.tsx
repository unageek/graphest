import {
  AlphaSlider,
  Caption1,
  ColorArea,
  ColorPicker,
  ColorSlider,
  Divider,
  Label,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  renderSwatchPickerGrid,
  SpinButton,
  SwatchPicker,
  Tab,
  TabList,
  TabValue,
  Text,
  ToolbarButton,
} from "@fluentui/react-components";
import { SharedColors } from "@fluentui/theme";
import * as Color from "color";
import * as React from "react";
import { MAX_PEN_SIZE } from "../common/constants";

export interface GraphStyleButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
  onPenSizeChanged: (penSize: number) => void;
  penSize: number;
}

export const GraphStyleButton = (props: GraphStyleButtonProps): JSX.Element => {
  const [selectedTab, setSelectedTab] = React.useState<TabValue>("swatch");

  return (
    <Popover positioning="below-start">
      <PopoverTrigger>
        <ToolbarButton
          title="Graph style"
          icon={
            <span
              style={{
                backgroundColor: props.color,
                height: "20px",
                width: "20px",
              }}
            />
          }
        ></ToolbarButton>
      </PopoverTrigger>
      <PopoverSurface style={{ padding: 0 }}>{renderMenuList()}</PopoverSurface>
    </Popover>
  );

  function renderMenuList(): JSX.Element {
    const color = new Color(props.color);
    const id = colorToId.get(color.hex());

    return (
      <div
        style={{
          // Adjusted to the width of the `SwatchColorPicker`.
          alignItems: "flex-start",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-start",
          padding: "10px",
          rowGap: "10px",
          width: "300px",
        }}
      >
        <TabList
          selectedValue={selectedTab}
          onTabSelect={(_, { value }) => setSelectedTab(value)}
          style={{ margin: "-10px -10px 0 -10px" }}
        >
          <Tab value="swatch">Swatch</Tab>
          <Tab value="custom">Custom</Tab>
        </TabList>
        {selectedTab === "swatch" && (
          <SwatchPicker
            layout="grid"
            onSelectionChange={(_, { selectedSwatch }) => {
              if (selectedSwatch !== undefined) {
                const newColor = new Color(selectedSwatch).alpha(color.alpha());
                props.onColorChanged(newColor.toString());
              }
            }}
            selectedValue={id}
            size="extra-small"
          >
            {renderSwatchPickerGrid({
              items: colorCells,
              columnCount: 12,
            })}
          </SwatchPicker>
        )}
        {selectedTab === "custom" && (
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
            <ColorArea />
            <ColorSlider />
            <AlphaSlider />
          </ColorPicker>
        )}
        <Divider />
        <div
          style={{
            alignItems: "baseline",
            display: "flex",
            flexDirection: "row",
          }}
        >
          <Label style={{ marginRight: "8px" }}>Pen size:</Label>
          <SpinButton
            defaultValue={props.penSize}
            max={MAX_PEN_SIZE}
            min={0}
            step={0.1}
            style={{ marginRight: "4px", width: "80px" }}
            onChange={(_, { value }) => {
              if (value === undefined) return;
              const penSize = Number(value);
              props.onPenSizeChanged(penSize);
            }}
          />
          <Text>{props.penSize > 1 ? "pixels" : "pixel"}</Text>
        </div>
        {props.penSize < 1.0 && (
          <Caption1>
            A pen size less than 1px is only applied to exported images.
          </Caption1>
        )}
        {props.penSize > 3.0 && (
          <Caption1>
            A pen size greater then 3px is only applied to exported images.
          </Caption1>
        )}
      </div>
    );
  }
};

const colorCells = [
  SharedColors.pinkRed10,
  SharedColors.red20,
  SharedColors.red10,
  SharedColors.redOrange20,
  SharedColors.redOrange10,
  SharedColors.orange30,
  SharedColors.orange20,
  SharedColors.orange10,
  SharedColors.yellow10,
  SharedColors.orangeYellow20,
  SharedColors.orangeYellow10,
  SharedColors.yellowGreen10,
  SharedColors.green20,
  SharedColors.green10,
  SharedColors.greenCyan10,
  SharedColors.cyan40,
  SharedColors.cyan30,
  SharedColors.cyan20,
  SharedColors.cyan10,
  SharedColors.cyanBlue20,
  SharedColors.cyanBlue10,
  SharedColors.blue10,
  SharedColors.blueMagenta40,
  SharedColors.blueMagenta30,
  SharedColors.blueMagenta20,
  SharedColors.blueMagenta10,
  SharedColors.magenta20,
  SharedColors.magenta10,
  SharedColors.magentaPink20,
  SharedColors.magentaPink10,
  SharedColors.gray40,
  SharedColors.gray30,
  SharedColors.gray20,
  SharedColors.gray10,
].map((c, i) => ({
  color: new Color(c).hex(),
  value: i.toString(),
}));

const colorToId: Map<string, string> = new Map(
  colorCells.map((c) => [c.color, c.value])
);
