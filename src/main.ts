import * as assert from "assert";
import { BigNumber } from "bignumber.js";
import { ChildProcess, execFile } from "child_process";
import { app, BrowserWindow, ipcMain, shell } from "electron";
import { autoUpdater } from "electron-updater";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { pathToFileURL } from "url";
import { BASE_ZOOM_LEVEL, GRAPH_TILE_SIZE } from "./constants";
import * as ipc from "./ipc";

const fsPromises = fs.promises;

// Lifecycle of a job
//
//        +----------+        +----------+        +----------+
//   ●--->|  Queued  +------->|  Active  +------->|  Exited  |
//        +----------+        +--+----+--+        +----------+
//                               | Λ  |
//                               | |  +--------------+
//                       SIGSTOP | | SIGCONT         | SIGKILL
//                               V |                 V
//                            +----+-----+        +----------+
//                            | Sleeping +------->| Aborted  |
//                            +----------+  SIG-  +----------+
//                                          KILL

/** The maximum number of running (both active and sleeping) jobs. */
const MAX_JOBS = 32;

/** The maximum number of active jobs. */
const MAX_ACTIVE_JOBS = 4;

/** The maximum amount of memory in MiB that each running job can use. */
const JOB_MEM_LIMIT = 64;

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

let queuedJobs: Job[] = [];
let activeJobs: Job[] = [];
let sleepingJobs: Job[] = [];

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

app.whenReady().then(() => {
  createWindow();
  autoUpdater.checkForUpdatesAndNotify();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("quit", () => {
  abortJobs();
  fs.rmdirSync(baseOutDir, { recursive: true });
});

app.on("window-all-closed", () => {
  app.quit();
});

ipcMain.handle<ipc.AbortGraphing>(
  ipc.abortGraphing,
  async (_, relId, tileId) => {
    abortJobs(
      (j) => j.relId === relId && (tileId === undefined || j.tileId === tileId)
    );
  }
);

ipcMain.handle<ipc.AbortGraphingAll>(ipc.abortGraphingAll, async () => {
  abortJobs();
});

ipcMain.handle<ipc.NewRelation>(ipc.newRelation, async (_, rel) => {
  const relId = nextRelId.toString();
  nextRelId++;
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
    const retina_scale = 2;
    if (tile === undefined) {
      const purturb_x = bignum(
        0.50123456789012345 / (retina_scale * GRAPH_TILE_SIZE)
      );
      const purturb_y = bignum(
        0.50234567890123456 / (retina_scale * GRAPH_TILE_SIZE)
      );
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
          (retina_scale * GRAPH_TILE_SIZE).toString(),
          (retina_scale * GRAPH_TILE_SIZE).toString(),
          "--dilate",
          (retina_scale - 1).toString(),
          "--gray-alpha",
          "--output",
          outFile,
          "--mem-limit",
          JOB_MEM_LIMIT.toString(),
        ],
        outFile,
        relId,
        tileId,
      };
      pushJob(job);
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

function abortJobs(filter: JobFilter = () => true) {
  const jobsToAbort = queuedJobs
    .concat(activeJobs, sleepingJobs)
    .filter(filter);
  for (const job of jobsToAbort) {
    const proc = job.proc;
    if (proc !== undefined && proc.exitCode === null) {
      proc.kill("SIGKILL");
    }
    job.aborted = true;
    relations.get(job.relId)?.tiles.delete(job.tileId);
    popJob(job);
  }
  updateQueue();
}

function bignum(x: number) {
  return new BigNumber(x);
}

function checkAndNotifyGraphingStatusChanged(relId: string) {
  const nJobs = countJobs((j) => j.relId === relId);
  if (nJobs === 1) {
    notifyGraphingStatusChanged(relId, true);
  } else if (nJobs === 0) {
    notifyGraphingStatusChanged(relId, false);
  }
}

function countJobs(filter: JobFilter = () => true): number {
  return (
    queuedJobs.filter(filter).length +
    activeJobs.filter(filter).length +
    sleepingJobs.filter(filter).length
  );
}

function deprioritize(job: Job) {
  if (activeJobs.length <= MAX_ACTIVE_JOBS && sleepingJobs.length === 0) {
    return;
  }

  const nBefore = countJobs();
  activeJobs = activeJobs.filter((j) => j !== job);
  sleepingJobs = sleepingJobs.filter((j) => j !== job);
  job.proc?.kill("SIGSTOP");
  sleepingJobs.push(job);
  const nAfter = countJobs();
  assert(nBefore === nAfter);

  updateQueue();
}

function makePromise(proc: ChildProcess): Promise<string | undefined> {
  return new Promise((resolve) => {
    proc.once("exit", (code) => {
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

function popJob(job: Job) {
  const nBefore = countJobs();
  queuedJobs = queuedJobs.filter((j) => j !== job);
  activeJobs = activeJobs.filter((j) => j !== job);
  sleepingJobs = sleepingJobs.filter((j) => j !== job);
  const nAfter = countJobs();
  assert(nBefore === nAfter + 1);
}

function pushJob(job: Job) {
  queuedJobs.push(job);
}

function updateQueue() {
  while (activeJobs.length < MAX_ACTIVE_JOBS && sleepingJobs.length > 0) {
    const job = sleepingJobs.shift();
    if (job !== undefined) {
      job.proc?.kill("SIGCONT");
      activeJobs.push(job);
    }
  }

  while (
    activeJobs.length + sleepingJobs.length < MAX_JOBS &&
    queuedJobs.length > 0
  ) {
    const job = queuedJobs.shift();
    if (job !== undefined) {
      // Don't do `await fsPromises.open(...)` here,
      // which can break the order of execution of `requestTile`/`abortGraphing`.
      fs.closeSync(fs.openSync(job.outFile, "w"));

      const onFileChange = () => {
        if (!job.aborted) {
          deprioritize(job);
          notifyTileReady(job.relId, job.tileId, true);
        }
      };

      let watcher: fs.FSWatcher | undefined;
      try {
        watcher = fs.watch(job.outFile, { persistent: false });
        watcher.on("change", onFileChange);
      } catch {
        // It is likely that the file has been deleted.
      }

      job.proc = execFile(graphExec, job.args);
      job.proc.once("exit", () => {
        job.proc = undefined;
        watcher?.off("change", onFileChange);
        watcher?.close();
        if (!job.aborted) {
          popJob(job);
          updateQueue();
          // This is required because neither the file stat may not have been updated
          // nor `watcher` may not have fired the 'change' event at this moment (why?).
          notifyTileReady(job.relId, job.tileId, true);
        }
        checkAndNotifyGraphingStatusChanged(job.relId);
      });

      const nBefore = countJobs();
      if (activeJobs.length < MAX_ACTIVE_JOBS) {
        activeJobs.push(job);
      } else {
        job.proc?.kill("SIGSTOP");
        sleepingJobs.unshift(job);
      }
      const nAfter = countJobs();
      assert(nAfter === nBefore + 1);

      checkAndNotifyGraphingStatusChanged(job.relId);
    }
  }
}
