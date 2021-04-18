import { BigNumber } from "bignumber.js";
import { ChildProcess, execFile } from "child_process";
import { app, BrowserWindow, ipcMain, shell } from "electron";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { pathToFileURL } from "url";
import { BASE_ZOOM_LEVEL, GRAPH_TILE_SIZE } from "./constants";
import * as ipc from "./ipc";

const fsPromises = fs.promises;

// The lifecycle of jobs:
//
//   +-------------+
//   |   Queued    |
//   +------O------+
//          |
//          V
//   +-------------+   SIGSTOP   +-------------+
//   |   Active    O------------>|  Suspended  |
//   |             |<------------O             |
//   +-----O-O-----+   SIGCONT   +------O------+
//         | |                          |
//         | +--------------------------O  SIGKILL
//         V                            V
//   +-------------+             +-------------+
//   |     Done    |             |   Aborted   |
//   +-------------+             +-------------+

/** The maximum number of active and suspended jobs. */
const MAX_JOBS = 32;
/** The maximum number of active jobs. */
const MAX_ACTIVE_JOBS = 6;
/** The maximum amount of RAM in MiB that each job can use. */
const MAX_RAM_PER_JOB = 64;

interface Job {
  aborted: boolean;
  args: string[];
  outFile: string;
  proc?: ChildProcess;
  relId: string;
  tileId: string;
}

type JobFilter = (job: Job) => boolean;

interface Tile {
  id: string;
  url: string;
  version?: number;
}

interface Relation {
  id: string;
  nextTileNumber: number;
  outDir: string;
  rel: string;
  tiles: Map<string, Tile>;
}

let activeJobs: Job[] = [];
let suspendedJobs: Job[] = [];
let queuedJobs: Job[] = [];

const baseOutDir: string = fs.mkdtempSync(path.join(os.tmpdir(), "graphest-"));
const graphExec: string = path.join(__dirname, "graph");
let mainWindow: BrowserWindow | undefined;
let nextRelId = 0;
const relations = new Map<string, Relation>();

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      spellcheck: false,
    },
  }).on("closed", () => {
    mainWindow = undefined;
  });
  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(createWindow);

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("quit", () => {
  abortGraphing();
  fs.rmdirSync(baseOutDir, { recursive: true });
});

app.on("window-all-closed", () => {
  app.quit();
});

ipcMain.handle<ipc.AbortGraphing>(
  ipc.abortGraphing,
  async (_, relId, tileId) => {
    abortGraphing(
      (job) =>
        job.relId === relId && (tileId === undefined || job.tileId === tileId)
    );
  }
);

ipcMain.handle<ipc.AbortGraphingAll>(ipc.abortGraphingAll, async () => {
  abortGraphing();
});

ipcMain.handle<ipc.NewRelation>(ipc.newRelation, async (_, rel) => {
  const relId = (nextRelId++).toString();
  const outDir = path.join(baseOutDir, relId);
  await fsPromises.mkdir(outDir);
  relations.set(relId, {
    id: relId,
    nextTileNumber: 0,
    outDir,
    rel,
    tiles: new Map(),
  });
  return { relId };
});

ipcMain.handle<ipc.OpenUrl>(ipc.openUrl, async (_, url) => {
  if (!url.startsWith("https://")) return;
  shell.openExternal(url);
});

ipcMain.handle<ipc.RequestTile>(
  ipc.requestTile,
  async (_, relId, tileId, coords) => {
    const rel = relations.get(relId);
    if (rel === undefined) {
      return;
    }

    const tile = rel.tiles.get(tileId);
    if (tile === undefined) {
      const purturb_x = bignum(0.50123456789012345 / GRAPH_TILE_SIZE);
      const purturb_y = bignum(0.50234567890123456 / GRAPH_TILE_SIZE);
      const widthPerTile = bignum(2 ** (BASE_ZOOM_LEVEL - coords.z));
      const x0 = widthPerTile.times(bignum(coords.x).minus(purturb_x));
      const x1 = widthPerTile.times(bignum(coords.x + 1).minus(purturb_x));
      const y0 = widthPerTile.times(bignum(-coords.y - 1).plus(purturb_y));
      const y1 = widthPerTile.times(bignum(-coords.y).plus(purturb_y));

      const outFile = path.join(rel.outDir, rel.nextTileNumber + ".png");
      rel.nextTileNumber++;

      const newTile: Tile = {
        id: tileId,
        url: pathToFileURL(outFile).href,
      };
      rel.tiles.set(tileId, newTile);

      const job: Job = {
        aborted: false,
        args: [
          rel.rel,
          "--bounds",
          x0.toString(),
          x1.toString(),
          y0.toString(),
          y1.toString(),
          "--size",
          GRAPH_TILE_SIZE.toString(),
          GRAPH_TILE_SIZE.toString(),
          "--gray-alpha",
          "--output",
          outFile,
          "--mem-limit",
          MAX_RAM_PER_JOB.toString(),
        ],
        outFile,
        relId,
        tileId,
      };
      queuedJobs.push(job);
      updateQueue();
    } else {
      if (tile.version !== undefined) {
        notifyTileReady(relId, tileId, false);
      }
    }
  }
);

ipcMain.handle<ipc.ValidateRelation>(ipc.validateRelation, async (_, rel) => {
  const error = await makePromise(execFile(graphExec, [rel, "--parse"]));
  return { error };
});

function abortGraphing(filter: JobFilter = () => true) {
  const jobsToAbort = activeJobs
    .concat(suspendedJobs, queuedJobs)
    .filter(filter);
  activeJobs = activeJobs.filter((j) => !filter(j));
  suspendedJobs = suspendedJobs.filter((j) => !filter(j));
  queuedJobs = queuedJobs.filter((j) => !filter(j));

  // Set all `aborted` flags true before `updateQueue` is called by killing any process.
  for (const job of jobsToAbort) {
    job.aborted = true;
  }

  for (const job of jobsToAbort) {
    const proc = job.proc;
    if (proc !== undefined && proc.exitCode === null) {
      proc.kill("SIGKILL");
    }
    relations.get(job.relId)?.tiles.delete(job.tileId);
  }
}

function bignum(x: number) {
  return new BigNumber(x);
}

async function deprioritize(job: Job) {
  if (activeJobs.length <= MAX_ACTIVE_JOBS && suspendedJobs.length === 0) {
    return;
  }
  activeJobs = activeJobs.filter((j) => j !== job);
  job.proc?.kill("SIGSTOP");
  suspendedJobs.push(job);
  await updateQueue();
}

async function dequeue(job: Job) {
  activeJobs = activeJobs.filter((j) => j !== job);
  suspendedJobs = suspendedJobs.filter((j) => j !== job);
  await updateQueue();

  const relId = job.relId;
  if (
    activeJobs.concat(suspendedJobs).filter((j) => j.relId === relId).length ===
    0
  ) {
    notifyGraphingStatusChanged(relId, false);
  }
}

function enqueue(job: Job) {
  if (activeJobs.length < MAX_ACTIVE_JOBS) {
    activeJobs.push(job);
  } else {
    job.proc?.kill("SIGSTOP");
    suspendedJobs.unshift(job);
  }

  const relId = job.relId;
  if (
    activeJobs.concat(suspendedJobs).filter((j) => j.relId === relId).length ===
    1
  ) {
    notifyGraphingStatusChanged(relId, true);
  }
}

function makePromise(proc: ChildProcess): Promise<string | undefined> {
  return new Promise(function (resolve) {
    proc.once("exit", function (code) {
      if (code === 0) {
        resolve(undefined);
      } else {
        // TODO: Return a meaningful error message.
        resolve("");
      }
    });
  });
}

function notifyGraphingStatusChanged(relId: string, processing: boolean) {
  mainWindow?.webContents.send<ipc.GraphingStatusChanged>(
    ipc.graphingStatusChanged,
    relId,
    processing
  );
}

function notifyTileReady(
  relId: string,
  tileId: string,
  incrementVersion: boolean
) {
  const tile = relations.get(relId)?.tiles.get(tileId);
  if (tile === undefined) {
    return;
  }

  if (incrementVersion) {
    tile.version = (tile.version ?? 0) + 1;
  }

  mainWindow?.webContents.send<ipc.TileReady>(
    ipc.tileReady,
    relId,
    tileId,
    tile.url + "?v=" + tile.version
  );
}

async function updateQueue() {
  while (activeJobs.length < MAX_ACTIVE_JOBS && suspendedJobs.length > 0) {
    const job = suspendedJobs.shift();
    if (job !== undefined) {
      job.proc?.kill("SIGCONT");
      activeJobs.push(job);
    }
  }

  while (
    activeJobs.length + suspendedJobs.length < MAX_JOBS &&
    queuedJobs.length > 0
  ) {
    const job = queuedJobs.shift();
    if (job !== undefined) {
      const f = await fsPromises.open(job.outFile, "w");
      await f.close();

      const onFileChange = function () {
        if (!job.aborted) {
          deprioritize(job);
          notifyTileReady(job.relId, job.tileId, true);
        }
      };

      job.proc = execFile(graphExec, job.args);
      job.proc.once("exit", () => {
        job.proc = undefined;
        dequeue(job);
        watcher?.off("change", onFileChange);
        watcher?.close();
        if (!job.aborted) {
          // This is required because neither the file stat may not have been updated
          // nor `watcher` may not have fired the 'change' event at this moment (why?).
          notifyTileReady(job.relId, job.tileId, true);
        }
      });
      enqueue(job);

      let watcher: fs.FSWatcher | undefined;
      try {
        watcher = fs.watch(job.outFile, { persistent: false });
        watcher.on("change", onFileChange);
      } catch {
        // It is likely that the file has been deleted.
      }
    }
  }
}
