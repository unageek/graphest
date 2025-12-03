import { bundleIcon, FluentIconsProps, wrapIcon } from "@fluentui/react-icons";
import { ReactNode, useMemo } from "react";

export interface ColorWellProps {
  color: string;
}

export const ColorWell = (props: ColorWellProps): ReactNode => {
  const { color } = props;

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
          <rect x="0" y="0" width="20" height="20" fill={color} />
        </svg>
      );
    });

    return bundleIcon(icon, icon);
  }, [color]);

  return <Icon />;
};
