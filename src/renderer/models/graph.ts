import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Graph {
  color: string;
  id: string;
  isProcessing: boolean;
  relation: string;
  relationInputByUser: boolean;
  relId: string;
  thickness: number;
}

const initialState: Graph = {
  color: "",
  id: "",
  isProcessing: false,
  relation: "",
  relationInputByUser: false,
  relId: "",
  thickness: 1,
};

const slice = createSlice({
  name: "graph",
  initialState,
  reducers: {
    setGraphColor: {
      prepare: (id: string, color: string) => ({ payload: { id, color } }),
      reducer: (s, a: PayloadAction<{ id: string; color: string }>) =>
        a.payload.id === s.id
          ? {
              ...s,
              color: a.payload.color,
            }
          : s,
    },
    setGraphIsProcessing: {
      prepare: (id: string, processing: boolean) => ({
        payload: { id, processing },
      }),
      reducer: (s, a: PayloadAction<{ id: string; processing: boolean }>) =>
        a.payload.id === s.id
          ? {
              ...s,
              isProcessing: a.payload.processing,
            }
          : s,
    },
    setGraphRelation: {
      prepare: (
        id: string,
        relId: string,
        rel: string,
        inputByUser: boolean
      ) => ({
        payload: { id, relId, rel, inputByUser },
      }),
      reducer: (
        s,
        a: PayloadAction<{
          id: string;
          relId: string;
          rel: string;
          inputByUser: boolean;
        }>
      ) =>
        a.payload.id === s.id
          ? {
              ...s,
              relation: a.payload.rel,
              relationInputByUser: a.payload.inputByUser,
              relId: a.payload.relId,
            }
          : s,
    },
    setGraphThickness: {
      prepare: (id: string, thickness: number) => ({
        payload: { id, thickness },
      }),
      reducer: (s, a: PayloadAction<{ id: string; thickness: number }>) =>
        a.payload.id === s.id
          ? {
              ...s,
              thickness: a.payload.thickness,
            }
          : s,
    },
  },
});

export const {
  setGraphColor,
  setGraphIsProcessing,
  setGraphRelation,
  setGraphThickness,
} = slice.actions;

export const graphReducer = slice.reducer;
