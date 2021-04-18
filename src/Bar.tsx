import { Stack, useTheme } from "@fluentui/react";
import * as React from "react";

export interface BarProps extends React.HTMLAttributes<HTMLElement> {
  padding?: number | string;
}

export const Bar = (props: BarProps): JSX.Element => {
  const theme = useTheme();

  return (
    <Stack
      horizontal
      styles={{
        root: {
          background: theme.semanticColors.bodyBackground,
          minHeight: "32px",
        },
      }}
      tokens={{
        padding: props.padding,
      }}
      {...props}
    />
  );
};
