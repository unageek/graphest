import { Tab, TabList, TabValue } from "@fluentui/react-components";
import * as React from "react";
import { CustomColorPicker } from "./CustomColorPicker";
import { SwatchColorPicker } from "./SwatchColorPicker";

export interface ColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorPicker = (props: ColorPickerProps): JSX.Element => {
  const [selectedTab, setSelectedTab] = React.useState<TabValue>("swatch");

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
      <TabList
        selectedValue={selectedTab}
        onTabSelect={(_, { value }) => setSelectedTab(value)}
        style={{ margin: "-12px -12px 4px -12px" }}
      >
        <Tab value="swatch">Swatch</Tab>
        <Tab value="custom">Custom</Tab>
      </TabList>
      {selectedTab === "swatch" && (
        <SwatchColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
      )}
      {selectedTab === "custom" && (
        <CustomColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
      )}
    </div>
  );
};
