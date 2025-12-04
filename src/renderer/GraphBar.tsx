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
import { ReactNode, useCallback, useRef } from "react";
import { useDispatch } from "react-redux";
import { RequestRelationResult } from "../common/ipc";
import { PenButton } from "./PenButton";
import { RelationInput, RelationInputActions } from "./RelationInput";
import { SymbolsButton } from "./SymbolsButton";
import { removeGraph, useSelector } from "./models/app";
import {
  setGraphColor,
  setGraphRelation,
  setGraphThickness,
} from "./models/graph";

export interface GraphBarProps {
  dragHandleProps?: DraggableProvidedDragHandleProps | null;
  focusGraphView: () => void;
  graphId: string;
  requestRelation: (
    rel: string,
    graphId: string,
  ) => Promise<RequestRelationResult>;
}

export const GraphBar = (props: GraphBarProps): ReactNode => {
  const { dragHandleProps, focusGraphView, graphId, requestRelation } = props;
  const dispatch = useDispatch();
  const graph = useSelector((s) => s.graphs.byId[props.graphId]);
  const relationInputActionsRef = useRef<RelationInputActions>(null);

  const onRelationChanged = useCallback(
    (relId: string, rel: string) => {
      dispatch(setGraphRelation(graphId, relId, rel, true));
    },
    [dispatch, graphId],
  );

  const requestRelationInner = useCallback(
    (rel: string) => {
      return requestRelation(rel, graphId);
    },
    [graphId, requestRelation],
  );

  return (
    <Toolbar style={{ background: tokens.colorNeutralBackground1 }}>
      <ToolbarButton
        {...dragHandleProps}
        appearance="transparent"
        icon={<ReOrderDotsVerticalRegular />}
        style={{
          color: tokens.colorNeutralForeground4,
          // https://github.com/hello-pangea/dnd/issues/711
          cursor: "grab",
        }}
        title="Drag to reorder"
      />
      <PenButton
        color={graph.color}
        onColorChanged={(c) => dispatch(setGraphColor(graphId, c))}
        onThicknessChanged={(t) => dispatch(setGraphThickness(graphId, t))}
        thickness={graph.thickness}
      />
      <RelationInput
        actionsRef={relationInputActionsRef}
        graphId={graphId}
        grow
        onEnterKeyPressed={focusGraphView}
        onRelationChanged={onRelationChanged}
        processing={graph.isProcessing}
        relation={graph.relation}
        relationInputByUser={graph.relationInputByUser}
        requestRelation={requestRelationInner}
      />
      <SymbolsButton
        onDismissed={() => relationInputActionsRef.current?.focus()}
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
              onClick={() => dispatch(removeGraph(graphId))}
            >
              Remove
            </MenuItem>
          </MenuList>
        </MenuPopover>
      </Menu>
    </Toolbar>
  );
};
