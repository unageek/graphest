import { CommandBarButton, Separator } from "@fluentui/react";
import * as React from "react";
import { useDispatch } from "react-redux";
import { Bar } from "./Bar";
import { newGraph, setShowGrid, useSelector } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();
  const showGrid = useSelector((s) => s.showGrid);

  return (
    <Bar>
      <CommandBarButton
        iconProps={{ iconName: "Add" }}
        onClick={() => dispatch(newGraph())}
        text="Add Relation"
      />
      <Separator vertical />
      <CommandBarButton
        checked={showGrid}
        iconProps={{ iconName: "GridViewMedium" }}
        onClick={() => dispatch(setShowGrid(!showGrid))}
        text="Show Grid"
        toggle
      />
    </Bar>
  );
};
