import {
  ColorPicker,
  DefaultButton,
  IColorCellProps,
  Pivot,
  PivotItem,
  Stack,
  SwatchColorPicker,
} from "@fluentui/react";
import { SharedColors } from "@fluentui/theme";
import * as Color from "color";
import * as React from "react";

export interface ColorButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorButton = (props: ColorButtonProps): JSX.Element => {
  return (
    <DefaultButton
      menuProps={{
        items: [{ key: "colors" }],
        onRenderMenuList: renderMenuList,
      }}
      styles={{
        root: {
          minWidth: 0,
          padding: 0,
          width: "52px",
        },
      }}
      title="Color"
    >
      <div
        style={{
          backgroundColor: props.color,
          height: "16px",
          width: "16px",
        }}
      >
        &nbsp; {/* For vertical alignment. */}
      </div>
    </DefaultButton>
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
        <Pivot>
          <PivotItem headerText="Swatch">
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
          </PivotItem>
          <PivotItem headerText="Custom">
            <ColorPicker
              alphaType="none"
              color={props.color}
              onChange={(_, c) => props.onColorChanged(c.str)}
              showPreview={true}
              styles={{
                panel: { padding: 0 },
                root: { margin: "8px" },
              }}
            />
          </PivotItem>
        </Pivot>
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
