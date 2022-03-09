import { ITextFieldProps, TextField } from "@fluentui/react";
import * as React from "react";

interface SendableTextFieldProps extends ITextFieldProps {
  onSend: () => void;
}

export const SendableTextField = (
  props: SendableTextFieldProps
): JSX.Element => {
  return (
    <TextField
      {...props}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          props.onSend();
        }
      }}
    />
  );
};
