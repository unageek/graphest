import {
  MenuButton,
  Popover,
  PopoverSurface,
  PopoverTrigger,
} from "@fluentui/react-components";
import * as React from "react";
import { ColorPicker } from "./ColorPicker";

export interface ColorButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorButton = (props: ColorButtonProps): JSX.Element => {
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
      <PopoverSurface>
        <ColorPicker
          color={props.color}
          onColorChanged={props.onColorChanged}
        />
      </PopoverSurface>
    </Popover>
  );
};
