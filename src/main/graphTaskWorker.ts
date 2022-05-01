import { execFile } from "child_process";
import { parentPort, workerData } from "worker_threads";
import { GraphTask } from "./graphTask";

const { abortController, args, executable }: GraphTask = workerData;

execFile(
  executable,
  args,
  {
    signal: abortController.signal,
  },
  (error, stdout, stderr) => {
    if (error !== null) {
      throw { ...error, stderr, stdout };
    }

    parentPort?.postMessage({ stderr, stdout });
  }
);
