import { tokens, Toolbar, ToolbarButton } from "@fluentui/react-components";
import {
  AddRegular,
  ArrowForwardRegular,
  ColorRegular,
  ImageRegular,
} from "@fluentui/react-icons";
import { ReactNode } from "react";
import { useDispatch } from "react-redux";
import {
  newGraph,
  setShowColorsDialog,
  setShowExportImageDialog,
  setShowGoToDialog,
} from "./models/app";

export const CommandBar = (): ReactNode => {
  const dispatch = useDispatch();

  return (
    <Toolbar style={{ background: tokens.colorNeutralBackground1 }}>
      <ToolbarButton icon={<AddRegular />} onClick={() => dispatch(newGraph())}>
        Add Relation
      </ToolbarButton>
      <ToolbarButton
        icon={<ColorRegular />}
        onClick={() => dispatch(setShowColorsDialog(true))}
      >
        Colors…
      </ToolbarButton>
      <ToolbarButton
        icon={<ArrowForwardRegular />}
        onClick={() => dispatch(setShowGoToDialog(true))}
      >
        Go To…
      </ToolbarButton>
      <ToolbarButton
        icon={<ImageRegular />}
        onClick={() => dispatch(setShowExportImageDialog(true))}
      >
        Render…
      </ToolbarButton>
    </Toolbar>
  );
};
