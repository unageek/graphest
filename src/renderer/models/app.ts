import { createSlice, isAnyOf, PayloadAction } from "@reduxjs/toolkit";
import { TypedUseSelectorHook, useSelector as _useSelector } from "react-redux";
import {
  BASE_ZOOM_LEVEL,
  DEFAULT_COLOR,
  INITIAL_ZOOM_LEVEL,
} from "../../common/constants";
import {
  ExportImageOptions,
  ExportImageProgress,
} from "../../common/exportImage";
import { GraphData } from "../graphData";
import {
  Graph,
  graphReducer,
  setGraphColor,
  setGraphIsProcessing,
  setGraphPenSize,
  setGraphRelation,
} from "./graph";

export interface AppState {
  center: [number, number];
  exportImageProgress: ExportImageProgress;
  graphs: { byId: { [id: string]: Graph }; allIds: string[] };
  highRes: boolean;
  lastExportImageOpts: ExportImageOptions;
  nextGraphId: number;
  resetView: boolean;
  showAxes: boolean;
  showExportImageDialog: boolean;
  showMajorGrid: boolean;
  showMinorGrid: boolean;
  zoomLevel: number;
}

const initialState: AppState = {
  center: [0, 0],
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
  resetView: false,
  showAxes: true,
  showExportImageDialog: false,
  showMajorGrid: true,
  showMinorGrid: true,
  zoomLevel: INITIAL_ZOOM_LEVEL - BASE_ZOOM_LEVEL,
};

const slice = createSlice({
  name: "app",
  initialState,
  reducers: {
    addGraph: {
      prepare: (data: GraphData) => ({
        payload: data,
      }),
      reducer: (s, a: PayloadAction<GraphData>) => {
        const { color, penSize, relation } = a.payload;
        const id = s.nextGraphId.toString();
        return {
          ...s,
          graphs: {
            byId: {
              ...s.graphs.byId,
              [id]: {
                color,
                id,
                isProcessing: false,
                penSize,
                relationInputByUser: false,
                relation,
                relId: "",
              },
            },
            allIds: [...s.graphs.allIds, id],
          },
          nextGraphId: s.nextGraphId + 1,
        };
      },
    },
    newGraph: (s) => {
      const id = s.nextGraphId.toString();
      return {
        ...s,
        graphs: {
          byId: {
            ...s.graphs.byId,
            [id]: {
              color: DEFAULT_COLOR,
              id,
              isProcessing: false,
              penSize: 1,
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
    removeAllGraphs: (s) => {
      return {
        ...s,
        graphs: { byId: {}, allIds: [] },
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
    setCenter: {
      prepare: (center: [number, number]) => ({ payload: { center } }),
      reducer: (s, a: PayloadAction<{ center: [number, number] }>) => {
        return {
          ...s,
          center: a.payload.center,
        };
      },
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
    setResetView: {
      prepare: (reset: boolean) => ({ payload: { reset } }),
      reducer: (s, a: PayloadAction<{ reset: boolean }>) => ({
        ...s,
        resetView: a.payload.reset,
      }),
    },
    setShowAxes: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showAxes: a.payload.show,
      }),
    },
    setShowExportImageDialog: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showExportImageDialog: a.payload.show,
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
    setZoomLevel: {
      prepare: (zoom: number) => ({ payload: { zoom } }),
      reducer: (s, a: PayloadAction<{ zoom: number }>) => {
        return {
          ...s,
          zoomLevel: a.payload.zoom,
        };
      },
    },
  },
  extraReducers: (builder) => {
    builder
      .addMatcher(
        isAnyOf(
          setGraphColor,
          setGraphIsProcessing,
          setGraphPenSize,
          setGraphRelation
        ),
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
  addGraph,
  newGraph,
  removeAllGraphs,
  removeGraph,
  reorderGraph,
  setCenter,
  setExportImageProgress,
  setHighRes,
  setLastExportImageOpts,
  setResetView,
  setShowAxes,
  setShowExportImageDialog,
  setShowMajorGrid,
  setShowMinorGrid,
  setZoomLevel,
} = slice.actions;

export const appReducer = slice.reducer;

export const useSelector: TypedUseSelectorHook<AppState> = _useSelector;
