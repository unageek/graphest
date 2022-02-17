import { createSlice, isAnyOf, PayloadAction } from "@reduxjs/toolkit";
import { TypedUseSelectorHook, useSelector as _useSelector } from "react-redux";
import {
  ExportImageOptions,
  ExportImageProgress,
} from "../../common/exportImage";
import {
  Graph,
  graphReducer,
  setGraphColor,
  setGraphIsProcessing,
  setGraphRelation,
} from "./graph";

export interface AppState {
  exportImageProgress: ExportImageProgress;
  graphs: { byId: { [id: string]: Graph }; allIds: string[] };
  highRes: boolean;
  lastExportImageOpts: ExportImageOptions;
  nextGraphId: number;
  showAxes: boolean;
  showExportDialog: boolean;
  showMajorGrid: boolean;
  showMinorGrid: boolean;
}

const initialState: AppState = {
  exportImageProgress: {
    lastStderr: "",
    lastUrl: "",
    progress: 0,
  },
  graphs: { byId: {}, allIds: [] },
  highRes: false,
  lastExportImageOpts: {
    antiAliasing: 5,
    height: 1024,
    path: "",
    timeout: 1000,
    width: 1024,
    xMax: "10",
    xMin: "-10",
    yMax: "10",
    yMin: "-10",
  },
  nextGraphId: 0,
  showAxes: true,
  showExportDialog: false,
  showMajorGrid: true,
  showMinorGrid: true,
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
              // `SharedColors.cyanBlue20`
              color: "rgba(0, 78, 140, 0.8)",
              id,
              isProcessing: false,
              relationInputByUser: false,
              relation: "y = sin(x)",
              relId: "",
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
    setExportImageProgress: {
      prepare: (progress: ExportImageProgress) => ({
        payload: { progress },
      }),
      reducer: (s, a: PayloadAction<{ progress: ExportImageProgress }>) => ({
        ...s,
        exportImageProgress: a.payload.progress,
      }),
    },
    setHighRes: {
      prepare: (highRes: boolean) => ({ payload: { highRes } }),
      reducer: (s, a: PayloadAction<{ highRes: boolean }>) => ({
        ...s,
        highRes: a.payload.highRes,
      }),
    },
    setLastExportImageOpts: {
      prepare: (opts: ExportImageOptions) => ({ payload: { opts } }),
      reducer: (s, a: PayloadAction<{ opts: ExportImageOptions }>) => ({
        ...s,
        lastExportImageOpts: a.payload.opts,
      }),
    },
    setShowAxes: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showAxes: a.payload.show,
      }),
    },
    setShowExportDialog: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showExportDialog: a.payload.show,
      }),
    },
    setShowMajorGrid: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showMajorGrid: a.payload.show,
      }),
    },
    setShowMinorGrid: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showMinorGrid: a.payload.show,
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
  setExportImageProgress,
  setHighRes,
  setLastExportImageOpts,
  setShowExportDialog,
  setShowAxes,
  setShowMajorGrid,
  setShowMinorGrid,
} = slice.actions;

export const appReducer = slice.reducer;

export const useSelector: TypedUseSelectorHook<AppState> = _useSelector;
