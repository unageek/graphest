import { ExecFileException } from "child_process";

export interface ExecFileWorkerArgs {
  args: string[];
  executable: string;
}

export interface ExecFileWorkerResult {
  stderr: string;
  stdout: string;
}

export type ExecFileWorkerError = ExecFileException & ExecFileWorkerResult;
