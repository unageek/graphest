import { CommandBarButton, Separator } from "@fluentui/react";
import * as React from "react";
import { useDispatch } from "react-redux";
import { Bar } from "./Bar";
import { newGraph, setShowGoToDialog } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();

  return (
    <Bar>
      <CommandBarButton
        iconProps={{ iconName: "Add" }}
        onClick={() => dispatch(newGraph())}
        text="Add Relation"
      />
      <Separator vertical />
      <CommandBarButton
        iconProps={{ iconName: "Forward" }}
        onClick={() => dispatch(setShowGoToDialog(true))}
        text="Go To…"
      />
    </Bar>
  );
};
