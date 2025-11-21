import {
  Toolbar,
  ToolbarButton,
  ToolbarDivider,
} from "@fluentui/react-components";
import {
  AddRegular,
  ArrowForwardRegular,
  ColorRegular,
} from "@fluentui/react-icons";
import * as React from "react";
import { useDispatch } from "react-redux";
import { newGraph, setShowColorsDialog, setShowGoToDialog } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();

  return (
    <Toolbar>
      <ToolbarButton icon={<AddRegular />} onClick={() => dispatch(newGraph())}>
        Add Relation
      </ToolbarButton>
      <ToolbarDivider />
      <ToolbarButton
        icon={<ColorRegular />}
        onClick={() => dispatch(setShowColorsDialog(true))}
      >
        Colors…
      </ToolbarButton>
      <ToolbarDivider />
      <ToolbarButton
        icon={<ArrowForwardRegular />}
        onClick={() => dispatch(setShowGoToDialog(true))}
      >
        Go To…
      </ToolbarButton>
    </Toolbar>
  );
};
