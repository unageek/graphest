import Color from "color";
import { z } from "zod";
import {
  BASE_ZOOM_LEVEL,
  DEFAULT_PEN_COLOR,
  INITIAL_ZOOM_LEVEL,
  MAX_PEN_THICKNESS,
} from "./constants";

const graphSchema = z.object({
  color: z
    .string()
    .transform((s) => {
      try {
        new Color(s);
        return s;
      } catch {
        return DEFAULT_PEN_COLOR;
      }
    })
    .default(DEFAULT_PEN_COLOR),
  relation: z.string().default(""),
  show: z.boolean().default(true),
  thickness: z
    .number()
    .transform((x) => Math.min(Math.max(x, 0), MAX_PEN_THICKNESS))
    .default(1),
});

export const documentSchema = z.object({
  background: z
    .string()
    .transform((s) => {
      try {
        new Color(s);
        return s;
      } catch {
        return "#ffffff";
      }
    })
    .default("#ffffff"),
  center: z.array(z.number()).length(2).default([0, 0]),
  foreground: z
    .string()
    .transform((s) => {
      try {
        new Color(s);
        return s;
      } catch {
        return "#000000";
      }
    })
    .default("#000000"),
  graphs: z.array(graphSchema).default([]),
  version: z.number(),
  zoomLevel: z
    .number()
    .int()
    .default(INITIAL_ZOOM_LEVEL - BASE_ZOOM_LEVEL),
});

export type Document = z.infer<typeof documentSchema>;
export type GraphData = z.infer<typeof graphSchema>;
