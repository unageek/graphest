import { useTheme } from "@fluentui/react";
import * as React from "react";
import {
  DragDropContext,
  Draggable,
  Droppable,
  DropResult,
} from "react-beautiful-dnd";
import { useDispatch } from "react-redux";
import { GraphBar } from "./GraphBar";
import { reorderGraph, useSelector } from "./models/app";

export interface GraphBarsProps {
  focusGraphView: () => void;
}

export const GraphBars = (props: GraphBarsProps): JSX.Element => {
  const dispatch = useDispatch();
  const graphs = useSelector((s) => s.graphs);
  const theme = useTheme();

  function onDragEnd(result: DropResult) {
    if (
      result.destination !== undefined &&
      result.destination.index !== result.source.index
    ) {
      dispatch(reorderGraph(result.source.index, result.destination.index));
    }
  }

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
                        ? theme.effects.elevation8
                        : undefined,
                      ...provided.draggableProps.style,
                    }}
                  >
                    <GraphBar
                      dragHandleProps={provided.dragHandleProps}
                      focusGraphView={props.focusGraphView}
                      graphId={id}
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
