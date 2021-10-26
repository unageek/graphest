import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Graph {
  color: string;
  id: string;
  isProcessing: boolean;
  relation: string;
  relationInputByUser: boolean;
}

const initialState: Graph = {
  color: "",
  id: "",
  isProcessing: false,
  relation: "",
  relationInputByUser: false,
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
      prepare: (id: string, relation: string, inputByUser: boolean) => ({
        payload: { id, relation, inputByUser: inputByUser },
      }),
      reducer: (
        s,
        a: PayloadAction<{ id: string; relation: string; inputByUser: boolean }>
      ) =>
        a.payload.id === s.id
          ? {
              ...s,
              relation: a.payload.relation,
              relationInputByUser: a.payload.inputByUser,
            }
          : s,
    },
  },
});

export const { setGraphColor, setGraphIsProcessing, setGraphRelation } =
  slice.actions;

export const graphReducer = slice.reducer;
