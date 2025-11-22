import {
  MenuButton,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  Tab,
  TabList,
  TabValue,
} from "@fluentui/react-components";
import * as React from "react";
import { MyColorPicker } from "./MyColorPicker";
import { MySwatchPicker } from "./MySwatchPicker";

export interface ColorButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorButton = (props: ColorButtonProps): JSX.Element => {
  const [selectedTab, setSelectedTab] = React.useState<TabValue>("swatch");

  return (
    <Popover positioning="below-start">
      <PopoverTrigger>
        <MenuButton
          style={{
            minWidth: "unset",
            width: "60px",
          }}
          title="Color"
        >
          <div
            style={{
              backgroundColor: props.color,
              height: "20px",
              width: "20px",
            }}
          >
            &nbsp; {/* For vertical alignment. */}
          </div>
        </MenuButton>
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
      </div>
    );
  }
};
