import { CommandBarButton, Separator } from "@fluentui/react";
import * as React from "react";
import { useDispatch } from "react-redux";
import { Bar } from "./Bar";
import { newGraph } from "./models/app";

export interface GraphCommandBarProps {
  showGoToDialog: () => void;
}

export const GraphCommandBar = (props: GraphCommandBarProps): JSX.Element => {
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
        onClick={() => props.showGoToDialog()}
        text="Go To…"
      />
    </Bar>
  );
};
