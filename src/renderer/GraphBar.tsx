import { Icon, useTheme } from "@fluentui/react";
import * as React from "react";
import { useRef } from "react";
import { DraggableProvidedDragHandleProps } from "react-beautiful-dnd";
import { useDispatch } from "react-redux";
import { RequestRelationResult } from "../common/ipc";
import { Bar } from "./Bar";
import { BarIconButton } from "./BarIconButton";
import { GraphStyleButton } from "./GraphStyleButton";
import { removeGraph, useSelector } from "./models/app";
import {
  setGraphColor,
  setGraphRelation,
  setGraphThickness,
} from "./models/graph";
import { RelationInput, RelationInputActions } from "./RelationInput";
import { SymbolsButton } from "./SymbolsButton";

export interface GraphBarProps {
  dragHandleProps?: DraggableProvidedDragHandleProps;
  focusGraphView: () => void;
  graphId: string;
  requestRelation: (
    rel: string,
    graphId: string,
    highRes: boolean
  ) => Promise<RequestRelationResult>;
}

export const GraphBar = (props: GraphBarProps): JSX.Element => {
  const dispatch = useDispatch();
  const graph = useSelector((s) => s.graphs.byId[props.graphId]);
  const highRes = useSelector((s) => s.highRes);
  const theme = useTheme();
  const relationInputActionsRef = useRef<RelationInputActions>(null);

  return (
    <Bar>
      <div
        style={{
          alignItems: "center",
          color: theme.semanticColors.disabledBodyText,
          display: "flex",
          justifyContent: "center",
          minWidth: "32px",
        }}
        {...props.dragHandleProps}
        title="Drag to move"
      >
        <Icon iconName="GripperDotsVertical" />
      </div>
      <GraphStyleButton
        color={graph.color}
        onColorChanged={(c) => dispatch(setGraphColor(props.graphId, c))}
        onThicknessChanged={(t) =>
          dispatch(setGraphThickness(props.graphId, t))
        }
        thickness={graph.thickness}
      />
      <RelationInput
        actionsRef={relationInputActionsRef}
        graphId={props.graphId}
        grow
        highRes={highRes}
        onEnterKeyPressed={props.focusGraphView}
        onRelationChanged={(relId, rel) =>
          dispatch(setGraphRelation(props.graphId, relId, rel, true))
        }
        processing={graph.isProcessing}
        relation={graph.relation}
        relationInputByUser={graph.relationInputByUser}
        requestRelation={(rel: string, highRes: boolean) => {
          return props.requestRelation(rel, props.graphId, highRes);
        }}
      />
      <SymbolsButton
        onSymbolChosen={(symbol: string) =>
          relationInputActionsRef.current?.insertSymbol(symbol)
        }
        onSymbolPairChosen={(left: string, right: string) =>
          relationInputActionsRef.current?.insertSymbolPair(left, right)
        }
      />
      <BarIconButton
        iconProps={{ iconName: "More" }}
        menuProps={{
          items: [
            {
              key: "remove",
              text: "Remove",
              iconProps: { iconName: "Delete" },
              onClick: () => {
                dispatch(removeGraph(props.graphId));
              },
            },
          ],
        }}
        styles={{
          menuIcon: { display: "none" },
          root: {
            marginRight: "8px",
          },
        }}
        title="Actions"
      />
    </Bar>
  );
};
