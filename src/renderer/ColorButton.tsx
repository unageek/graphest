import {
  Button,
  Popover,
  PopoverSurface,
  PopoverTrigger,
} from "@fluentui/react-components";
import { ReactNode } from "react";
import { ColorPicker } from "./ColorPicker";
import { ColorWell } from "./ColorWell";

export interface ColorButtonProps {
  color: string;
  onColorChanged: (color: string) => void;
}

export const ColorButton = (props: ColorButtonProps): ReactNode => {
  return (
    <Popover positioning="below-start" trapFocus={true}>
      <PopoverTrigger>
        <Button icon={<ColorWell color={props.color} />} title="Color"></Button>
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
