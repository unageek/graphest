import { CommandBarButton, IButtonProps } from "@fluentui/react";
import * as React from "react";

export const BarIconButton = (props: IButtonProps): JSX.Element => {
  return (
    <CommandBarButton
      {...props}
      styles={{
        ...props.styles,
        root: {
          minWidth: "36px",
          ...(typeof props.styles?.root === "object"
            ? props.styles?.root
            : undefined),
          padding: "0 4px",
        },
      }}
    />
  );
};
