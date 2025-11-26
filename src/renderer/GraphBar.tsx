import {
  Menu,
  MenuItem,
  MenuList,
  MenuPopover,
  MenuTrigger,
  tokens,
  Toolbar,
  ToolbarButton,
} from "@fluentui/react-components";
import {
  DeleteRegular,
  MoreHorizontalRegular,
  ReOrderDotsVerticalRegular,
} from "@fluentui/react-icons";
import { DraggableProvidedDragHandleProps } from "@hello-pangea/dnd";
import * as React from "react";
import { useRef } from "react";
import { useDispatch } from "react-redux";
import { RequestRelationResult } from "../common/ipc";
import { GraphStyleButton } from "./GraphStyleButton";
import { RelationInput, RelationInputActions } from "./RelationInput";
import { SymbolsButton } from "./SymbolsButton";
import { removeGraph, useSelector } from "./models/app";
import {
  setGraphColor,
  setGraphPenSize,
  setGraphRelation,
} from "./models/graph";

export interface GraphBarProps {
  dragHandleProps?: DraggableProvidedDragHandleProps | null;
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
  const relationInputActionsRef = useRef<RelationInputActions>(null);

  return (
    <Toolbar style={{ background: tokens.colorNeutralBackground1 }}>
      <div
        style={{
          alignItems: "center",
          color: tokens.colorNeutralForeground4,
          // https://github.com/hello-pangea/dnd/issues/711
          cursor: "grab",
          display: "flex",
          fontSize: "20px",
          justifyContent: "center",
          minWidth: "32px",
        }}
        {...props.dragHandleProps}
        title="Drag to reorder"
      >
        <ReOrderDotsVerticalRegular />
      </div>
      <GraphStyleButton
        color={graph.color}
        onColorChanged={(c) => dispatch(setGraphColor(props.graphId, c))}
        onPenSizeChanged={(t) => dispatch(setGraphPenSize(props.graphId, t))}
        penSize={graph.penSize}
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
      <Menu>
        <MenuTrigger>
          <ToolbarButton icon={<MoreHorizontalRegular />} />
        </MenuTrigger>
        <MenuPopover>
          <MenuList>
            <MenuItem
              icon={<DeleteRegular />}
              onClick={() => {
                dispatch(removeGraph(props.graphId));
              }}
            >
              Remove
            </MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </Toolbar>
  );
};
