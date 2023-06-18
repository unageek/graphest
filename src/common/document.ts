import * as Color from "color";
import { z } from "zod";
import {
  BASE_ZOOM_LEVEL,
  DEFAULT_COLOR,
  INITIAL_ZOOM_LEVEL,
  MAX_PEN_SIZE,
} from "./constants";

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
  graphs: z
    .array(
      z.object({
        color: z
          .string()
          .transform((s) => {
            try {
              new Color(s);
              return s;
            } catch {
              return DEFAULT_COLOR;
            }
          })
          .default(DEFAULT_COLOR),
        penSize: z
          .number()
          .transform((x) => Math.min(Math.max(x, 0), MAX_PEN_SIZE))
          .default(1),
        relation: z.string().default(""),
      })
    )
    .default([]),
  version: z.number(),
  zoomLevel: z
    .number()
    .int()
    .default(INITIAL_ZOOM_LEVEL - BASE_ZOOM_LEVEL),
});

export type Document = z.infer<typeof documentSchema>;
