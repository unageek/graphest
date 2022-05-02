import { execFile } from "child_process";
import { parentPort, workerData } from "worker_threads";
import { ExecFileWorkerArgs } from "./execFileWorkerInterfaces";

const { args, executable }: ExecFileWorkerArgs = workerData;

// Follow the behavior of `promisify(execFile)`:
//   https://github.com/nodejs/node/blob/f8ca5dfea462d05c4fadd6a935f375a7aa71f8be/lib/child_process.js#L227
execFile(executable, args, (error, stdout, stderr) => {
  if (error !== null) {
    throw { ...error, stderr, stdout };
  }

  parentPort?.postMessage({ stderr, stdout });
});
