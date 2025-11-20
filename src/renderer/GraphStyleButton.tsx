import {
  IColorCellProps,
  Label,
  Separator,
  SpinButton,
  Stack,
  SwatchColorPicker,
  Text,
} from "@fluentui/react";
import {
  AlphaSlider,
  ColorArea,
  ColorPicker,
  ColorSlider,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  Tab,
  TabList,
  TabValue,
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
                height: "16px",
                width: "16px",
              }}
            />
          }
        ></ToolbarButton>
      </PopoverTrigger>
      <PopoverSurface>{renderMenuList()}</PopoverSurface>
    </Popover>
  );

  function renderMenuList(): JSX.Element {
    const color = new Color(props.color);
    const id = colorToId.get(color.hex());

    return (
      <Stack
        style={{
          // Adjusted to the width of the `SwatchColorPicker`.
          width: "288px",
        }}
      >
        <TabList
          selectedValue={selectedTab}
          onTabSelect={(_, { value }) => setSelectedTab(value)}
        >
          <Tab value="swatch">Swatch</Tab>
          <Tab value="custom">Custom</Tab>
        </TabList>
        <>
          {selectedTab === "swatch" && (
            <SwatchColorPicker
              cellMargin={8}
              cellShape={"square"}
              colorCells={colorCells}
              columnCount={10}
              onChange={(_, __, c) => {
                if (c !== undefined) {
                  const newColor = new Color(c).alpha(color.alpha());
                  props.onColorChanged(newColor.toString());
                }
              }}
              selectedId={id}
              styles={{ root: { margin: "4px", padding: 0 } }}
            />
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
        </>
        <Separator styles={{ root: { height: "1px", padding: 0 } }} />
        <Stack style={{ margin: "8px" }}>
          <Stack horizontal verticalAlign="baseline">
            <Label style={{ marginRight: "8px" }}>Pen size:</Label>
            <SpinButton
              defaultValue={props.penSize.toString()}
              max={MAX_PEN_SIZE}
              min={0}
              step={0.1}
              styles={{ root: { marginRight: "4px", width: "50px" } }}
              onChange={(_, value) => {
                if (value === undefined) return;
                const penSize = Number(value);
                props.onPenSizeChanged(penSize);
              }}
            />
            <Text>pixels</Text>
          </Stack>
          {props.penSize < 1.0 && (
            <Text style={{ marginTop: "8px" }} variant="small">
              A pen size less than 1px is only applied to exported images.
            </Text>
          )}
          {props.penSize > 3.0 && (
            <Text style={{ marginTop: "8px" }} variant="small">
              A pen size greater then 3px is only applied to exported images.
            </Text>
          )}
        </Stack>
      </Stack>
    );
  }
};

const colorCells: IColorCellProps[] = [
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
  id: i.toString(),
  color: new Color(c).hex(),
}));

const colorToId: Map<string, string> = new Map(
  colorCells.map((c) => [c.color, c.id])
);
