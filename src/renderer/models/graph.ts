import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export interface Graph {
  color: string;
  id: string;
  relation: string;
  relationInputByUser: boolean;
  relId: string | null;
  show: boolean;
  thickness: number;
}

const initialState: Graph = {
  color: "",
  id: "",
  relation: "",
  relationInputByUser: false,
  relId: null,
  show: true,
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
    setGraphRelation: {
      prepare: (
        id: string,
        relId: string | null,
        rel: string,
        inputByUser: boolean,
      ) => ({
        payload: { id, relId, rel, inputByUser },
      }),
      reducer: (
        s,
        a: PayloadAction<{
          id: string;
          relId: string | null;
          rel: string;
          inputByUser: boolean;
        }>,
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
    setGraphShow: {
      prepare: (id: string, show: boolean) => ({ payload: { id, show } }),
      reducer: (s, a: PayloadAction<{ id: string; show: boolean }>) =>
        a.payload.id === s.id
          ? {
              ...s,
              show: a.payload.show,
            }
          : s,
    },
    setGraphThickness: {
      prepare: (id: string, thickness: number) => ({
        payload: { id, thickness: thickness },
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
  setGraphRelation,
  setGraphShow,
  setGraphThickness,
} = slice.actions;

export const graphReducer = slice.reducer;
