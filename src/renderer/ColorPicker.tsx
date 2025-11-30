import { Tab, TabList, TabValue } from "@fluentui/react-components";
import { ReactNode, useState } from "react";
import { CustomColorPicker } from "./CustomColorPicker";
import { SwatchColorPicker } from "./SwatchColorPicker";

export interface ColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorPicker = (props: ColorPickerProps): ReactNode => {
  const [selectedTab, setSelectedTab] = useState<TabValue>("swatch");

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        // Adjusted to the width of the `SwatchColorPicker`.
        width: "284px",
      }}
    >
      <TabList
        onTabSelect={(_, { value }) => setSelectedTab(value)}
        selectedValue={selectedTab}
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
