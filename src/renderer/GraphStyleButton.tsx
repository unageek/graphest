import {
  Caption1,
  Divider,
  Field,
  Input,
  Label,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  ToolbarButton,
} from "@fluentui/react-components";
import { debounce } from "lodash";
import * as React from "react";
import { useMemo, useState } from "react";
import { MAX_PEN_SIZE } from "../common/constants";
import { tryParseNumberInRange } from "../common/parse";
import { ColorPicker } from "./ColorPicker";

export interface GraphStyleButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
  onPenSizeChanged: (penSize: number) => void;
  penSize: number;
}

export const GraphStyleButton = (
  props: GraphStyleButtonProps,
): React.ReactNode => {
  const { onPenSizeChanged } = props;
  const [penSize, setPenSize] = useState<string>(props.penSize.toString());
  const [penSizeErrorMessage, setPenSizeErrorMessage] = useState<string>();

  const validatePenSize = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseNumberInRange(value, 0, MAX_PEN_SIZE);
        setPenSizeErrorMessage(result.err);
        if (result.ok !== undefined) {
          onPenSizeChanged(result.ok);
        }
      }, 200),
    [onPenSizeChanged],
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
        <div
          style={{
            alignItems: "baseline",
            display: "flex",
            flexDirection: "row",
            gap: "8px",
          }}
        >
          <Label style={{ textWrap: "nowrap" }}>Pen size:</Label>
          <Field validationMessage={penSizeErrorMessage}>
            <Input
              contentAfter={props.penSize > 1 ? "pixels" : "pixel"}
              onChange={(_, { value }) => {
                setPenSize(value);
                validatePenSize(value);
              }}
              style={{ width: "150px" }}
              value={penSize}
            />
          </Field>
        </div>
        {props.penSize < 1.0 && (
          <Caption1>
            A pen size less than 1px is only applied to rendered graphs.
          </Caption1>
        )}
        {props.penSize > 3.0 && (
          <Caption1>
            A pen size greater then 3px is only applied to rendered graphs.
          </Caption1>
        )}
      </div>
    );
  }
};
