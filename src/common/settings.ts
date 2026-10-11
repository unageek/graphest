import { z } from "zod";
import { SystemInformation } from "./ipc";

export const settingsSchema = z.object({
  cpuLimit: z.number().int().min(1).catch(4),
  memoryLimit: z.number().int().min(1).catch(2),
  renderingParallelism: z.number().int().min(1).optional().catch(undefined),
});

export type Settings = z.infer<typeof settingsSchema>;

/** Returns the default value for {@link Settings.renderingParallelism}. */
export function getDefaultRenderingParallelism(
  info: SystemInformation,
): number {
  return Math.min(info.logicalCores, Math.floor(info.memory / 2));
}

/** Returns the maximum value for {@link Settings.renderingParallelism}. */
export function getMaxRenderingParallelism(info: SystemInformation): number {
  return Math.min(info.logicalCores, info.memory);
}
