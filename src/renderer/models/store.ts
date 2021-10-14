import { configureStore } from "@reduxjs/toolkit";
import { appReducer, newGraph } from "./app";

export const store = configureStore({
  reducer: appReducer,
});

store.dispatch(newGraph());

export type Store = typeof store;
