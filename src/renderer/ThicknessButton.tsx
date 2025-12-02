import { ToggleButton, ToggleButtonProps } from "@fluentui/react-components";
import { bundleIcon, FluentIconsProps, wrapIcon } from "@fluentui/react-icons";
import { ReactNode, useMemo } from "react";

export interface ThicknessButtonProps {
  thickness: number;
}

export const ThicknessButton = (
  props: ThicknessButtonProps & ToggleButtonProps,
): ReactNode => {
  const { thickness } = props;

  const Icon = useMemo(() => {
    const icon = wrapIcon((props: FluentIconsProps) => {
      return (
        <svg
          height="1em"
          width="1em"
          role="presentation"
          focusable="false"
          viewBox="0 0 20 20"
          {...props}
        >
          <line
            x1="2"
            y1="18"
            x2="18"
            y2="2"
            stroke="currentColor"
            strokeLinecap="round"
            strokeWidth={thickness}
          />
        </svg>
      );
    });

    return bundleIcon(icon, icon);
  }, [thickness]);

  return <ToggleButton icon={<Icon />} {...props} />;
};
