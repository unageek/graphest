import {
  Caption1,
  Divider,
  Label,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  SpinButton,
  Tab,
  TabList,
  TabValue,
  Text,
  ToolbarButton,
} from "@fluentui/react-components";
import * as React from "react";
import { MAX_PEN_SIZE } from "../common/constants";
import { MyColorPicker } from "./MyColorPicker";
import { MySwatchPicker } from "./MySwatchPicker";

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
      <PopoverSurface style={{ padding: 0 }}>{renderPopover()}</PopoverSurface>
    </Popover>
  );

  function renderPopover(): JSX.Element {
    return (
      <div
        style={{
          // Adjusted to the width of the `SwatchColorPicker`.
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          padding: "10px",
          width: "284px",
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
          <MySwatchPicker
            color={props.color}
            onColorChanged={props.onColorChanged}
          />
        )}
        {selectedTab === "custom" && (
          <MyColorPicker
            color={props.color}
            onColorChanged={props.onColorChanged}
          />
        )}
        <Divider />
        <div
          style={{
            alignItems: "baseline",
            display: "flex",
            flexDirection: "row",
            gap: "8px",
          }}
        >
          <Label>Pen size:</Label>
          <SpinButton
            defaultValue={props.penSize}
            max={MAX_PEN_SIZE}
            min={0}
            step={0.1}
            style={{ width: "80px" }}
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
