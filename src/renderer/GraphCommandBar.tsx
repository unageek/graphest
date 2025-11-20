import {
  Toolbar,
  ToolbarButton,
  ToolbarDivider,
} from "@fluentui/react-components";
import { AddIcon, ColorIcon, ForwardIcon } from "@fluentui/react-icons-mdl2";
import * as React from "react";
import { useDispatch } from "react-redux";
import { newGraph, setShowColorsDialog, setShowGoToDialog } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();

  return (
    <Toolbar>
      <ToolbarButton icon={<AddIcon />} onClick={() => dispatch(newGraph())}>
        Add Relation
      </ToolbarButton>
      <ToolbarDivider />
      <ToolbarButton
        icon={<ColorIcon />}
        onClick={() => dispatch(setShowColorsDialog(true))}
      >
        Colors…
      </ToolbarButton>
      <ToolbarDivider />
      <ToolbarButton
        icon={<ForwardIcon />}
        onClick={() => dispatch(setShowGoToDialog(true))}
      >
        Go To…
      </ToolbarButton>
    </Toolbar>
  );
};
