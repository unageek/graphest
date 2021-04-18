import { CommandBarButton, Separator } from "@fluentui/react";
import * as React from "react";
import { useDispatch } from "react-redux";
import { Bar } from "./Bar";
import * as ipc from "./ipc";
import { newGraph, setShowGrid, useSelector } from "./models/app";

export const GraphCommandBar = (): JSX.Element => {
  const dispatch = useDispatch();
  const showGrid = useSelector((s) => s.showGrid);

  function showHelp() {
    window.ipcRenderer.invoke<ipc.OpenUrl>(
      ipc.openUrl,
      "https://github.com/unageek/graphest/blob/master/docs/guide/README.adoc"
    );
  }

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
      <Separator vertical />
      <CommandBarButton
        iconProps={{ iconName: "Unknown" }}
        onClick={() => showHelp()}
        text="Help"
      />
    </Bar>
  );
};
