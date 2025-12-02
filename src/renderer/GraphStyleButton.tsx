import {
  Caption1,
  Divider,
  Field,
  Input,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  ToolbarButton,
  ToolbarDivider,
} from "@fluentui/react-components";
import { debounce } from "lodash";
import { ReactNode, useMemo, useState } from "react";
import { MAX_PEN_SIZE } from "../common/constants";
import { tryParseNumberInRange } from "../common/parse";
import { ColorPicker } from "./ColorPicker";
import { ThicknessButton } from "./ThicknessButton";

export interface GraphStyleButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
  onThicknessChanged: (penSize: number) => void;
  thickness: number;
}

export const GraphStyleButton = (props: GraphStyleButtonProps): ReactNode => {
  const { onThicknessChanged } = props;
  const [thickness, setThickness] = useState<string>(
    props.thickness.toString(),
  );
  const [thicknessErrorMessage, setThicknessErrorMessage] = useState<string>();

  const validateThickness = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseNumberInRange(value, 0, MAX_PEN_SIZE);
        setThicknessErrorMessage(result.err);
        if (result.ok !== undefined) {
          onThicknessChanged(result.ok);
        }
      }, 200),
    [onThicknessChanged],
  );

  return (
    <Popover positioning="below-start">
      <PopoverTrigger>
        <ToolbarButton
          icon={
            <span
              style={{
                backgroundColor: props.color,
                height: "20px",
                width: "20px",
              }}
            />
          }
          title="Graph style"
        />
      </PopoverTrigger>
      <PopoverSurface>{renderPopover()}</PopoverSurface>
    </Popover>
  );

  function renderPopover() {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          // Adjusted to the width of the `ColorPicker`.
          width: "284px",
        }}
      >
        <ColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
        <Divider />
        <Field validationMessage={thicknessErrorMessage}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              flexDirection: "row",
            }}
          >
            {[1, 2, 3].map((t) => (
              <ThicknessButton
                appearance="subtle"
                checked={props.thickness === t}
                onClick={() => {
                  setThickness(t.toString());
                  onThicknessChanged(t);
                }}
                thickness={t}
              />
            ))}
            <ToolbarDivider />
            <Input
              contentAfter={props.thickness > 1 ? "pixels" : "pixel"}
              onChange={(_, { value }) => {
                setThickness(value);
                validateThickness(value);
              }}
              style={{ width: "120px" }}
              value={thickness}
            />
          </div>
        </Field>
        {props.thickness < 1.0 && (
          <Caption1>
            A pen thickness less than 1px is only applied to rendered graphs.
          </Caption1>
        )}
        {props.thickness > 3.0 && (
          <Caption1>
            A pen thickness greater then 3px is only applied to rendered graphs.
          </Caption1>
        )}
      </div>
    );
  }
};
