import { IStackProps, Stack, useTheme } from "@fluentui/react";
import * as React from "react";

export interface BarProps extends IStackProps {
  padding?: number | string;
}

export const Bar = (props: BarProps): JSX.Element => {
  const theme = useTheme();

  return (
    <Stack
      {...props}
      horizontal
      styles={{
        ...props.styles,
        root: {
          background: theme.semanticColors.bodyBackground,
          minHeight: "32px",
        },
      }}
      tokens={{
        padding: props.padding,
      }}
    />
  );
};
