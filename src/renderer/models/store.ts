import { configureStore } from "@reduxjs/toolkit";
import { appReducer } from "./app";

export const store = configureStore({
  reducer: appReducer,
});

export type Store = typeof store;
