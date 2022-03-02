import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Graph {
  color: string;
  id: string;
  isProcessing: boolean;
  penSize: number;
  relation: string;
  relationInputByUser: boolean;
  relId: string;
}

const initialState: Graph = {
  color: "",
  id: "",
  isProcessing: false,
  penSize: 1,
  relation: "",
  relationInputByUser: false,
  relId: "",
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
    setGraphPenSize: {
      prepare: (id: string, penSize: number) => ({
        payload: { id, penSize },
      }),
      reducer: (s, a: PayloadAction<{ id: string; penSize: number }>) =>
        a.payload.id === s.id
          ? {
              ...s,
              penSize: a.payload.penSize,
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
  },
});

export const {
  setGraphColor,
  setGraphIsProcessing,
  setGraphPenSize,
  setGraphRelation,
} = slice.actions;

export const graphReducer = slice.reducer;
