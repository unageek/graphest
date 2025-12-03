import { Tab, TabList } from "@fluentui/react-components";
import { ReactNode, useState } from "react";
import { CustomColorPicker } from "./CustomColorPicker";
import { SwatchColorPicker } from "./SwatchColorPicker";

export interface ColorPickerProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorPicker = (props: ColorPickerProps): ReactNode => {
  const [selectedTab, setSelectedTab] = useState<ColorPickerTab>(
    ColorPickerTab.Swatches,
  );

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
        onTabSelect={(_, { value }) => setSelectedTab(value as ColorPickerTab)}
        selectedValue={selectedTab}
        style={{ margin: "-12px -12px 4px -12px" }}
      >
        <Tab value={ColorPickerTab.Swatches}>Swatches</Tab>
        <Tab value={ColorPickerTab.Custom}>Custom</Tab>
      </TabList>
      {selectedTab === ColorPickerTab.Swatches && (
        <SwatchColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
      )}
      {selectedTab === ColorPickerTab.Custom && (
        <CustomColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
      )}
    </div>
  );
};

enum ColorPickerTab {
  Swatches = "swatches",
  Custom = "custom",
}
