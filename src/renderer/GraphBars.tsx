import { tokens } from "@fluentui/react-components";
import {
  DragDropContext,
  Draggable,
  Droppable,
  DropResult,
} from "@hello-pangea/dnd";
import * as React from "react";
import { useCallback } from "react";
import { useDispatch } from "react-redux";
import { RequestRelationResult } from "../common/ipc";
import { GraphBar } from "./GraphBar";
import { reorderGraph, useSelector } from "./models/app";

export interface GraphBarsProps {
  focusGraphView: () => void;
  requestRelation: (
    rel: string,
    graphId: string,
    highRes: boolean,
  ) => Promise<RequestRelationResult>;
}

export const GraphBars = (props: GraphBarsProps): React.ReactNode => {
  const dispatch = useDispatch();
  const graphs = useSelector((s) => s.graphs);

  const onDragEnd = useCallback(
    (result: DropResult) => {
      if (
        result.destination &&
        result.destination.index !== result.source.index
      ) {
        dispatch(reorderGraph(result.source.index, result.destination.index));
      }
    },
    [dispatch],
  );

  return (
    <DragDropContext onDragEnd={onDragEnd}>
      <Droppable droppableId="graphBars">
        {(provided) => (
          <div ref={provided.innerRef} {...provided.droppableProps}>
            {graphs.allIds.map((id, index) => (
              <Draggable key={id} draggableId={id.toString()} index={index}>
                {(provided, snapshot) => (
                  <div
                    ref={provided.innerRef}
                    {...provided.draggableProps}
                    style={{
                      boxShadow: snapshot.isDragging
                        ? tokens.shadow8
                        : undefined,
                      ...provided.draggableProps.style,
                    }}
                  >
                    <GraphBar
                      dragHandleProps={provided.dragHandleProps}
                      focusGraphView={props.focusGraphView}
                      graphId={id}
                      requestRelation={props.requestRelation}
                    />
                  </div>
                )}
              </Draggable>
            ))}
            {provided.placeholder}
          </div>
        )}
      </Droppable>
    </DragDropContext>
  );
};
