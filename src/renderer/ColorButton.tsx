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

export const ColorButton = (props: ColorButtonProps): React.ReactNode => {
  return (
    <Popover positioning="below-start">
      <PopoverTrigger>
        <MenuButton title="Color">
          <span
            style={{
              backgroundColor: props.color,
              height: "20px",
              width: "20px",
            }}
          >
            &nbsp; {/* For vertical alignment. */}
          </span>
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
