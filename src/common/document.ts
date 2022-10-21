import * as Color from "color";
import { array, InferType, number, object, string } from "yup";
import {
  BASE_ZOOM_LEVEL,
  DEFAULT_COLOR,
  INITIAL_ZOOM_LEVEL,
  MAX_PEN_SIZE,
} from "./constants";

export const documentSchema = object({
  background: string()
    .transform((s) => {
      try {
        new Color(s);
        return s;
      } catch {
        return "#ffffff";
      }
    })
    .default("#ffffff"),
  center: array(number()).length(2).default([0, 0]),
  foreground: string()
    .transform((s) => {
      try {
        new Color(s);
        return s;
      } catch {
        return "#000000";
      }
    })
    .default("#000000"),
  graphs: array(
    object({
      color: string()
        .transform((s) => {
          try {
            new Color(s);
            return s;
          } catch {
            return DEFAULT_COLOR;
          }
        })
        .default(DEFAULT_COLOR),
      penSize: number()
        .transform((x) => Math.min(Math.max(x, 0), MAX_PEN_SIZE))
        .default(1),
      relation: string().default(""),
    })
  ).required(),
  version: number().required(),
  zoomLevel: number()
    .integer()
    .default(INITIAL_ZOOM_LEVEL - BASE_ZOOM_LEVEL),
});

export type Document = InferType<typeof documentSchema>;
