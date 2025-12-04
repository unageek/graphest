import { tokens } from "@fluentui/react-components";
import {
  DragDropContext,
  Draggable,
  DragUpdate,
  Droppable,
  DropResult,
} from "@hello-pangea/dnd";
import { ReactNode, useCallback } from "react";
import { useDispatch } from "react-redux";
import { RequestRelationResult } from "../common/ipc";
import { GraphBar } from "./GraphBar";
import { reorderGraph, setGraphTransposition, useSelector } from "./models/app";

export interface GraphBarsProps {
  focusGraphView: () => void;
  requestRelation: (
    rel: string,
    graphId: string,
  ) => Promise<RequestRelationResult>;
}

export const GraphBars = (props: GraphBarsProps): ReactNode => {
  const dispatch = useDispatch();
  const graphs = useSelector((s) => s.graphs);

  const onDragEnd = useCallback(
    ({ source, destination }: DropResult) => {
      if (destination && destination.index !== source.index) {
        dispatch(reorderGraph(source.index, destination.index));
      }
      dispatch(setGraphTransposition(null));
    },
    [dispatch],
  );

  const onDragUpdate = useCallback(
    ({ source, destination }: DragUpdate) => {
      if (destination) {
        dispatch(setGraphTransposition([source.index, destination.index]));
      }
    },
    [dispatch],
  );

  return (
    <DragDropContext onDragEnd={onDragEnd} onDragUpdate={onDragUpdate}>
      <Droppable droppableId="graphBars">
        {(provided) => (
          <div ref={provided.innerRef} {...provided.droppableProps}>
            {graphs.allIds.map((id, index) => (
              <Draggable
                key={id}
                disableInteractiveElementBlocking
                draggableId={id.toString()}
                index={index}
              >
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
