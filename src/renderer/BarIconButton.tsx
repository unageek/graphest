import { CommandBarButton, IButtonProps } from "@fluentui/react";
import * as React from "react";

export const BarIconButton = (props: IButtonProps): JSX.Element => {
  return (
    <CommandBarButton
      {...props}
      styles={{
        ...props.styles,
        root: {
          ...(typeof props.styles?.root === "object"
            ? props.styles?.root
            : undefined),
          minWidth: "36px",
          padding: "0 4px",
        },
      }}
    />
  );
};
