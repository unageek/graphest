import {
  ColorPicker,
  IColorCellProps,
  Label,
  Pivot,
  PivotItem,
  Separator,
  SpinButton,
  Stack,
  SwatchColorPicker,
  Text,
} from "@fluentui/react";
import { SharedColors } from "@fluentui/theme";
import * as Color from "color";
import * as React from "react";
import { BarIconButton } from "./BarIconButton";

export interface GraphStyleButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
  onPenSizeChanged: (penSize: number) => void;
  penSize: number;
}

export const GraphStyleButton = (props: GraphStyleButtonProps): JSX.Element => {
  return (
    <BarIconButton
      menuProps={{
        items: [{ key: "colors" }],
        onRenderMenuList: () => renderMenuList(),
      }}
      styles={{
        menuIcon: { display: "none" },
      }}
      title="Style"
    >
      <div
        style={{
          backgroundColor: props.color,
          height: "16px",
          width: "16px",
        }}
      />
    </BarIconButton>
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
              styles={{ root: { margin: "4px", padding: "0" } }}
            />
          </PivotItem>
          <PivotItem headerText="Custom">
            <ColorPicker
              color={props.color}
              onChange={(_, c) => props.onColorChanged(c.str)}
              styles={{
                panel: { padding: "0" },
                root: { margin: "8px" },
              }}
            />
          </PivotItem>
        </Pivot>
        <Separator styles={{ root: { height: "1px", padding: "0" } }} />
        <Stack style={{ margin: "8px" }}>
          <Stack
            horizontal
            style={{ alignItems: "baseline", marginBottom: "8px" }}
          >
            <Label style={{ marginRight: "8px" }}>Pen size:</Label>
            <SpinButton
              defaultValue={props.penSize.toString()}
              max={1000}
              min={0}
              step={0.1}
              styles={{ root: { marginRight: "4px", width: "50px" } }}
              onChange={(_, v) => {
                if (v === undefined) return;
                const penSize = Number(v);
                if (Number.isFinite(penSize)) {
                  props.onPenSizeChanged(penSize);
                }
              }}
            />
            <Text>pixels</Text>
          </Stack>
          <Text variant="small">
            The pen size is only applied to exported images.
          </Text>
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
