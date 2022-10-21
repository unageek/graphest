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

export type ThemeName = "dark" | "light";

export interface AppState {
  center: [number, number];
  exportImageProgress: ExportImageProgress;
  graphBackground: string;
  graphForeground: string;
  graphs: { byId: { [id: string]: Graph }; allIds: string[] };
  highRes: boolean;
  lastExportImageOpts: ExportImageOptions;
  nextGraphId: number;
  resetView: boolean;
  showAxes: boolean;
  showColorsDialog: boolean;
  showExportImageDialog: boolean;
  showGoToDialog: boolean;
  showMajorGrid: boolean;
  showMinorGrid: boolean;
  theme: ThemeName;
  zoomLevel: number;
}

const initialState: AppState = {
  center: [0, 0],
  exportImageProgress: {
    messages: [],
    progress: 0,
  },
  graphBackground: "#ffffff",
  graphForeground: "#000000",
  graphs: { byId: {}, allIds: [] },
  highRes: false,
  lastExportImageOpts: {
    antiAliasing: 5,
    correctAlpha: false,
    height: 1024,
    path: "",
    timeout: 10,
    transparent: false,
    width: 1024,
    xMax: "10",
    xMin: "-10",
    yMax: "10",
    yMin: "-10",
  },
  nextGraphId: 0,
  resetView: false,
  showAxes: true,
  showColorsDialog: false,
  showExportImageDialog: false,
  showGoToDialog: false,
  showMajorGrid: true,
  showMinorGrid: true,
  theme: "light",
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
    setGraphBackground: {
      prepare: (color: string) => ({
        payload: { color },
      }),
      reducer: (s, a: PayloadAction<{ color: string }>) => ({
        ...s,
        graphBackground: a.payload.color,
      }),
    },
    setGraphForeground: {
      prepare: (color: string) => ({
        payload: { color },
      }),
      reducer: (s, a: PayloadAction<{ color: string }>) => ({
        ...s,
        graphForeground: a.payload.color,
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
    setShowColorsDialog: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showColorsDialog: a.payload.show,
      }),
    },
    setShowExportImageDialog: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showExportImageDialog: a.payload.show,
      }),
    },
    setShowGoToDialog: {
      prepare: (show: boolean) => ({ payload: { show } }),
      reducer: (s, a: PayloadAction<{ show: boolean }>) => ({
        ...s,
        showGoToDialog: a.payload.show,
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
    setTheme: {
      prepare: (theme: ThemeName) => ({ payload: { theme } }),
      reducer: (s, a: PayloadAction<{ theme: ThemeName }>) => ({
        ...s,
        theme: a.payload.theme,
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
  setGraphBackground,
  setGraphForeground,
  setHighRes,
  setLastExportImageOpts,
  setResetView,
  setShowAxes,
  setShowColorsDialog,
  setShowExportImageDialog,
  setShowGoToDialog,
  setShowMajorGrid,
  setShowMinorGrid,
  setTheme,
  setZoomLevel,
} = slice.actions;

export const appReducer = slice.reducer;

export const useSelector: TypedUseSelectorHook<AppState> = _useSelector;
