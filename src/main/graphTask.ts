export interface GraphTask {
  abortController: AbortController;
  args: string[];
  executable: string;
  outFile: string;
}
