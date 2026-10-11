import { z } from "zod";

export const settingsSchema = z.object({
  cpuLimit: z.number().int().min(1).catch(4),
  memoryLimit: z.number().int().min(1).catch(2),
});

export type Settings = z.infer<typeof settingsSchema>;
