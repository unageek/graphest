import { createSlice, isAnyOf, PayloadAction } from "@reduxjs/toolkit";
import { TypedUseSelectorHook, useSelector as _useSelector } from "react-redux";
import {
  Graph,
  graphReducer,
  setGraphColor,
  setGraphIsProcessing,
  setGraphRelation,
} from "./graph";

export interface AppState {
  graphs: { byId: { [id: string]: Graph }; allIds: string[] };
  highRes: boolean;
  nextGraphId: number;
  showAxes: boolean;
  showGrid: boolean;
}

const initialState: AppState = {
  graphs: { byId: {}, allIds: [] },
  highRes: false,
  nextGraphId: 0,
  showAxes: true,
  showGrid: true,
};

const slice = createSlice({
  name: "app",
  initialState,
  reducers: {
    newGraph: (s) => {
      const id = s.nextGraphId.toString();
      return {
        ...s,
        graphs: {
          byId: {
            ...s.graphs.byId,
            [id]: {
              color: "rgba(0, 0, 0, 0.75)",
              id,
              isProcessing: false,
              relation: "y = sin(x)",
              relationInputByUser: false,
            },
          },
          allIds: [...s.graphs.allIds, id],
        },
        nextGraphId: s.nextGraphId + 1,
      };
    },
    removeGraph: {
      prepare: (id: string) => ({ payload: { id } }),
      reducer: (s, a: PayloadAction<{ id: string }>) => ({
        ...s,
        graphs: {
          byId: Object.fromEntries(
            Object.entries(s.graphs.byId).filter(([id]) => id !== a.payload.id)
          ),
          allIds: s.graphs.allIds.filter((id) => id !== a.payload.id),
        },
      }),
    },
    reorderGraph: {
      prepare: (fromIndex: number, toIndex: number) => ({
        payload: { fromIndex, toIndex },
      }),
      reducer: (
        s,
        a: PayloadAction<{ fromIndex: number; toIndex: number }>
      ) => ({
        ...s,
        graphs: {
          ...s.graphs,
          allIds: moveElement(
            s.graphs.allIds,
            a.payload.fromIndex,
            a.payload.toIndex
          ),
        },
      }),
    },
    setHighRes: {
      prepare: (highRes: boolean) => ({ payload: { highRes } }),
      reducer: (s, a: PayloadAction<{ highRes: boolean }>) => ({
        ...s,
        highRes: a.payload.highRes,
      }),
    },
    setShowAxes: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showAxes: a.payload.show,
      }),
    },
    setShowGrid: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showGrid: a.payload.show,
      }),
    },
  },
  extraReducers: (builder) => {
    builder
      .addMatcher(
        isAnyOf(setGraphColor, setGraphIsProcessing, setGraphRelation),
        (s, a) => ({
          ...s,
          graphs: {
            ...s.graphs,
            byId: {
              ...s.graphs.byId,
              [a.payload.id]: graphReducer(s.graphs.byId[a.payload.id], a),
            },
          },
        })
      )
      .addDefaultCase(() => {
        return;
      });
  },
});

function moveElement<T>(array: T[], fromIndex: number, toIndex: number): T[] {
  const result = [...array];
  const [removed] = result.splice(fromIndex, 1);
  result.splice(toIndex, 0, removed);
  return result;
}

export const {
  newGraph,
  removeGraph,
  reorderGraph,
  setHighRes,
  setShowAxes,
  setShowGrid,
} = slice.actions;

export const appReducer = slice.reducer;

export const useSelector: TypedUseSelectorHook<AppState> = _useSelector;
