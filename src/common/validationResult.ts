import { Range } from "./range";

export interface ValidationError {
  range: Range;
  message: string;
}

export type ValidationResult = ValidationError | null;
