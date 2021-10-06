import { CommandBarButton } from "@fluentui/react";
import * as React from "react";
import { useDispatch } from "react-redux";
import { Bar } from "./Bar";
import { newGraph } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();

  return (
    <Bar>
      <CommandBarButton
        iconProps={{ iconName: "Add" }}
        onClick={() => dispatch(newGraph())}
        text="Add Relation"
      />
    </Bar>
  );
};
