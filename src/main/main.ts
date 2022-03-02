import * as assert from "assert";
import { ChildProcess, execFile } from "child_process";
import { app, BrowserWindow, dialog, ipcMain, Menu, shell } from "electron";
import installExtension, {
  REACT_DEVELOPER_TOOLS,
  REDUX_DEVTOOLS,
} from "electron-devtools-installer";
import { autoUpdater } from "electron-updater";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as url from "url";
import * as util from "util";
import { bignum } from "../common/bignumber";
import { Command } from "../common/command";
import {
  BASE_ZOOM_LEVEL,
  GRAPH_TILE_EXTENSION,
  GRAPH_TILE_SIZE,
  PERTURBATION_X,
  PERTURBATION_Y,
} from "../common/constants";
import {
  ANTI_ALIASING_OPTIONS,
  ExportImageEntry,
  ExportImageOptions,
  ExportImageProgress,
  EXPORT_GRAPH_TILE_SIZE,
  MAX_EXPORT_IMAGE_SIZE,
  MAX_EXPORT_TIMEOUT,
} from "../common/exportImage";
import * as ipc from "../common/ipc";
import { Range } from "../common/range";
import * as result from "../common/result";
import { createMainMenu } from "./mainMenu";

const fsPromises = fs.promises;

// The Lifecycle of a Job
//
//                        |  Running graph(1)  |
//                        |                    |
//                        |                    |
//        +----------+    |    +----------+    |    +----------+
//   ●--->|  Queued  +-------->|  Active  +-------->|  Exited  |
//        +----------+    |    +--+----+--+    |    +----------+
//                        |       | Λ  |       |
//                        |       | |  +---------------+
//                       Pause *1 | | Resume *2|       | SIGKILL
//                        |       V |          |       V
//                        |    +----+-----+    |    +----------+
//                        |    | Sleeping +-------->| Aborted  |
//                        |    +----------+  SIG-   +----------+
//                        |                  KILL
//                        |                    |
//
// *1. graph(1) pauses execution right after initialization and every after it finishes
//     writing to the output image.
// *2. graph(1) resumes execution when it receives a newline character from stdin.

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
  highRes: boolean;
  id: string;
  nextTileNumber: number;
  outDir: string;
  rel: string;
  tiles: Map<string, Tile>;
}

function getBundledExecutable(name: string): string {
  // ".exe" is required for pointing to executables inside .asar archives.
  return path.join(
    __dirname,
    process.platform === "win32" ? name + ".exe" : name
  );
}

let queuedJobs: Job[] = [];
let activeJobs: Job[] = [];
let sleepingJobs: Job[] = [];

const baseOutDir: string = fs.mkdtempSync(path.join(os.tmpdir(), "graphest-"));
const composeExec: string = getBundledExecutable("compose");
let exportImageAbortController: AbortController | undefined;
const graphExec: string = getBundledExecutable("graph");
const joinTilesExec: string = getBundledExecutable("join-tiles");
let mainMenu: Menu | undefined;
let mainWindow: BrowserWindow | undefined;
let nextExportImageId = 0;
let nextRelId = 0;
const relationById = new Map<string, Relation>();
const relKeyToRelId = new Map<string, string>();

function createMainWindow() {
  mainWindow = new BrowserWindow({
    height: 600,
    minHeight: 200,
    minWidth: 200,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      spellcheck: false,
    },
    width: 800,
  }).on("closed", () => {
    mainWindow = undefined;
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

function resetBrowserZoom() {
  // Another possible solution:
  //   https://github.com/electron/electron/issues/10572#issuecomment-944822575

  try {
    const prefsFile = path.join(app.getPath("userData"), "Preferences");
    const prefs = JSON.parse(fs.readFileSync(prefsFile, "utf8"));
    delete prefs.partition;
    fs.writeFileSync(prefsFile, JSON.stringify(prefs));
  } catch {
    // ignore
  }
}

app.whenReady().then(async () => {
  // https://github.com/MarshallOfSound/electron-devtools-installer/issues/195#issuecomment-932634933
  await installExtension([REACT_DEVELOPER_TOOLS, REDUX_DEVTOOLS], {
    loadExtensionOptions: { allowFileAccess: true },
  });

  resetBrowserZoom();
  mainMenu = createMainMenu({
    [Command.AbortGraphing]: () => abortJobs(),
    [Command.ExportImage]: () => {
      mainWindow?.webContents.send(ipc.commandInvoked, Command.ExportImage);
    },
    [Command.HighResolution]: () => {
      mainWindow?.webContents.send(ipc.commandInvoked, Command.HighResolution);
    },
    [Command.ShowAxes]: () => {
      mainWindow?.webContents.send(ipc.commandInvoked, Command.ShowAxes);
    },
    [Command.ShowMajorGrid]: () => {
      mainWindow?.webContents.send(ipc.commandInvoked, Command.ShowMajorGrid);
    },
    [Command.ShowMinorGrid]: () => {
      mainWindow?.webContents.send(ipc.commandInvoked, Command.ShowMinorGrid);
    },
  });
  Menu.setApplicationMenu(mainMenu);
  createMainWindow();
  autoUpdater.checkForUpdatesAndNotify();
});

app.on("quit", () => {
  abortJobs();
  fs.rmdirSync(baseOutDir, { recursive: true });
});

app.on("window-all-closed", () => {
  app.quit();
});

ipcMain.handle(ipc.abortExportImage, async () => {
  exportImageAbortController?.abort();
});

ipcMain.handle(ipc.abortGraphing, async (_, relId: string, tileId?: string) => {
  abortJobs(
    (j) => j.relId === relId && (tileId === undefined || j.tileId === tileId)
  );
});

ipcMain.handle(
  ipc.exportImage,
  async (
    _,
    entries: ExportImageEntry[],
    opts: ExportImageOptions
  ): Promise<void> => {
    const outDir = path.join(baseOutDir, "export");
    if (!fs.existsSync(outDir)) {
      await fsPromises.mkdir(outDir);
    }

    const bounds = [
      bignum(opts.xMin),
      bignum(opts.xMax),
      bignum(opts.yMin),
      bignum(opts.yMax),
    ];

    if (
      !(
        bounds.every((x) => x.isFinite()) &&
        bounds[0].lt(bounds[1]) &&
        bounds[2].lt(bounds[3]) &&
        ANTI_ALIASING_OPTIONS.includes(opts.antiAliasing) &&
        Number.isInteger(opts.height) &&
        opts.height > 0 &&
        opts.height <= MAX_EXPORT_IMAGE_SIZE &&
        Number.isInteger(opts.timeout) &&
        opts.timeout > 0 &&
        opts.timeout <= MAX_EXPORT_TIMEOUT &&
        Number.isInteger(opts.width) &&
        opts.width > 0 &&
        opts.width <= MAX_EXPORT_IMAGE_SIZE
      )
    ) {
      return;
    }

    const newEntries = [];
    for (const entry of entries) {
      newEntries.push({
        path: path.join(outDir, nextExportImageId.toString() + ".png"),
        tilePathPrefix: path.join(outDir, nextExportImageId.toString() + "-"),
        tilePathSuffix: ".png",
        ...entry,
      });
      nextExportImageId++;
    }

    const pixelWidth = bounds[1].minus(bounds[0]).div(opts.width);
    const pixelHeight = bounds[3].minus(bounds[2]).div(opts.height);
    const x0 = bounds[0].minus(pixelWidth.times(PERTURBATION_X));
    const y1 = bounds[3].minus(pixelHeight.times(PERTURBATION_Y));

    exportImageAbortController = new AbortController();
    const { signal } = exportImageAbortController;

    for (let k = 0; k < newEntries.length; k++) {
      const entry = newEntries[k];
      const rel = relationById.get(entry.relId);
      if (rel === undefined) {
        return;
      }

      const x_tiles = Math.ceil(
        (opts.antiAliasing * opts.width) / EXPORT_GRAPH_TILE_SIZE
      );
      const y_tiles = Math.ceil(
        (opts.antiAliasing * opts.height) / EXPORT_GRAPH_TILE_SIZE
      );
      const tile_width = Math.ceil(opts.width / x_tiles);
      const tile_height = Math.ceil(opts.height / y_tiles);
      for (let i_tile = 0; i_tile < y_tiles; i_tile++) {
        const i = i_tile * tile_height;
        const height = Math.min(tile_height, opts.height - i);
        for (let j_tile = 0; j_tile < x_tiles; j_tile++) {
          const j = j_tile * tile_width;
          const width = Math.min(tile_width, opts.width - j);

          const bounds = [
            x0.plus(pixelWidth.times(j)),
            x0.plus(pixelWidth.times(j + width)),
            y1.minus(pixelHeight.times(i + height)),
            y1.minus(pixelHeight.times(i)),
          ];

          const path = `${entry.tilePathPrefix}${i_tile}-${j_tile}${entry.tilePathSuffix}`;
          const args = [
            "--bounds",
            ...bounds.map((b) => b.toString()),
            "--gray-alpha",
            "--output",
            path,
            "--output-once",
            "--size",
            width.toString(),
            height.toString(),
            "--ssaa",
            opts.antiAliasing.toString(),
            "--thickness",
            entry.thickness.toString(),
            "--timeout",
            opts.timeout.toString(),
            "--",
            rel.rel,
          ];
          try {
            // Somehow, type definition is messed up!
            /* eslint-disable @typescript-eslint/no-explicit-any */
            const { stderr } = (await util.promisify(execFile)(
              graphExec,
              args,
              { signal } as any
            )) as unknown as { stdout: string; stderr: string };
            /* eslint-enable @typescript-eslint/no-explicit-any */
            if (stderr) {
              console.log(stderr.trimEnd());
            }
            notifyExportImageStatusChanged({
              lastStderr: stderr.trimEnd(),
              lastUrl: url.pathToFileURL(path).toString(),
              progress:
                (x_tiles * y_tiles * k + x_tiles * i_tile + j_tile + 1) /
                (x_tiles * y_tiles * newEntries.length),
            });
          } catch ({ name, stderr }) {
            if (typeof stderr !== "string") {
              throw new Error("unexpected type");
            }
            console.log(stderr.trimEnd());
            console.log("`graph` failed:", `'${args.join("' '")}'`);
            return;
          }
        }
      }

      const args = [
        "--output",
        entry.path,
        "--prefix",
        entry.tilePathPrefix,
        "--size",
        opts.width.toString(),
        opts.height.toString(),
        "--suffix",
        entry.tilePathSuffix,
        "--x-tiles",
        x_tiles.toString(),
        "--y-tiles",
        y_tiles.toString(),
      ];
      try {
        /* eslint-disable @typescript-eslint/no-explicit-any */
        const { stderr } = (await util.promisify(execFile)(
          joinTilesExec,
          args,
          { signal } as any
        )) as unknown as { stdout: string; stderr: string };
        /* eslint-enable @typescript-eslint/no-explicit-any */
        if (stderr) {
          console.log(stderr.trimEnd());
        }
      } catch ({ stderr }) {
        if (typeof stderr !== "string") {
          throw new Error("unexpected type");
        }
        console.log(stderr.trimEnd());
        console.log("`join-tiles` failed:", `'${args.join("' '")}'`);
        return;
      }
    }

    const args = [
      ...newEntries.flatMap((entry) => ["--add", entry.path, entry.color]),
      "--output",
      opts.path,
    ];
    try {
      /* eslint-disable @typescript-eslint/no-explicit-any */
      const { stderr } = (await util.promisify(execFile)(composeExec, args, {
        signal,
      } as any)) as unknown as { stdout: string; stderr: string };
      /* eslint-enable @typescript-eslint/no-explicit-any */
      if (stderr) {
        console.log(stderr.trimEnd());
      }
    } catch ({ stderr }) {
      if (typeof stderr !== "string") {
        throw new Error("unexpected type");
      }
      console.log(stderr.trimEnd());
      console.log("`compose` failed:", `'${args.join("' '")}'`);
      return;
    }

    await shell.openPath(opts.path);
  }
);

ipcMain.handle(ipc.getDefaultExportImagePath, async (): Promise<string> => {
  try {
    return path.join(app.getPath("pictures"), "graph.png");
  } catch {
    return "";
  }
});

ipcMain.handle(
  ipc.requestRelation,
  async (
    _,
    rel: string,
    graphId: string,
    highRes: boolean
  ): Promise<ipc.RequestRelationResult> => {
    try {
      const args = ["--parse", "--dump-ast", "--", rel];
      const { stdout } = await util.promisify(execFile)(graphExec, args);
      const ast = stdout.split("\n")[0];
      const relKey = graphId + ":" + highRes.toString() + ":" + ast;
      let relId = relKeyToRelId.get(relKey);
      if (relId === undefined) {
        relId = nextRelId.toString();
        nextRelId++;
        const outDir = path.join(baseOutDir, relId);
        await fsPromises.mkdir(outDir);
        relationById.set(relId, {
          highRes,
          id: relId,
          nextTileNumber: 0,
          outDir,
          rel,
          tiles: new Map(),
        });
        relKeyToRelId.set(relKey, relId);
      }
      return result.ok(relId);
    } catch ({ stderr }) {
      if (typeof stderr !== "string") {
        throw new Error("`stderr` must be a string");
      }
      const lines = stderr.split("\n");
      const start =
        parseInt((lines[1].match(/^.*:\d+:(\d+)/) as RegExpMatchArray)[1]) - 1;
      const len = (lines[3].match(/~*$/) as RegExpMatchArray)[0].length;
      const message = (lines[1].match(/error: (.*)$/) as RegExpMatchArray)[1];
      return result.err({
        range: new Range(start, start + len),
        message,
      });
    }
  }
);

ipcMain.handle(
  ipc.requestTile,
  async (_, relId: string, tileId: string, coords: L.Coords) => {
    const rel = relationById.get(relId);
    if (rel === undefined) {
      return;
    }

    const tile = rel.tiles.get(tileId);
    const retinaScale = rel.highRes ? 2 : 1;
    if (tile === undefined) {
      // We offset the graph by 0.5px to place the origin at the center of a pixel.
      // The direction of offsetting must be coherent with the configuration of `GridLayer`.
      // We also add asymmetric perturbation to the offset so that
      // points with simple coordinates may not be located on pixel boundaries,
      // which could make lines such as `x y = 0` or `(x + y)(x − y) = 0` look thicker.
      const pixelOffsetX = bignum(
        (0.5 + PERTURBATION_X) / (retinaScale * GRAPH_TILE_SIZE)
      );
      const pixelOffsetY = bignum(
        (0.5 + PERTURBATION_Y) / (retinaScale * GRAPH_TILE_SIZE)
      );
      const widthPerTile = bignum(2 ** (BASE_ZOOM_LEVEL - coords.z));
      const x0 = widthPerTile.times(bignum(coords.x).minus(pixelOffsetX));
      const x1 = widthPerTile.times(bignum(coords.x + 1).minus(pixelOffsetX));
      const y0 = widthPerTile.times(bignum(-coords.y - 1).plus(pixelOffsetY));
      const y1 = widthPerTile.times(bignum(-coords.y).plus(pixelOffsetY));

      const outFile = path.join(rel.outDir, rel.nextTileNumber + ".png");
      rel.nextTileNumber++;

      const newTile: Tile = {
        id: tileId,
        url: url.pathToFileURL(outFile).href,
      };
      rel.tiles.set(tileId, newTile);

      const job: Job = {
        aborted: false,
        args: [
          "--bounds",
          x0.toString(),
          x1.toString(),
          y0.toString(),
          y1.toString(),
          "--size",
          (retinaScale * GRAPH_TILE_SIZE).toString(),
          (retinaScale * GRAPH_TILE_SIZE).toString(),
          "--pad-right",
          (retinaScale * GRAPH_TILE_EXTENSION).toString(),
          "--pad-bottom",
          (retinaScale * GRAPH_TILE_EXTENSION).toString(),
          "--dilate",
          retinaScale === 2 ? "0,0,0;0,1,1;0,1,1" : "1",
          "--gray-alpha",
          "--output",
          outFile,
          "--mem-limit",
          JOB_MEM_LIMIT.toString(),
          "--pause-per-iteration",
          "--",
          rel.rel,
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

ipcMain.handle(
  ipc.showSaveDialog,
  async (_, path: string): Promise<string | undefined> => {
    if (!mainWindow) return undefined;
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: path,
      filters: [{ name: "PNG", extensions: ["png"] }],
    });
    return result.filePath;
  }
);

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
    relationById.get(job.relId)?.tiles.delete(job.tileId);
    popJob(job);
  }
  updateQueue();
}

function checkAndNotifyGraphingStatusChanged(relId: string) {
  const nJobs = countJobs();
  const abortGraphingMenu = mainMenu?.getMenuItemById(Command.AbortGraphing);
  if (abortGraphingMenu) {
    abortGraphingMenu.enabled = nJobs > 0;
  }

  const nRelJobs = countJobs((j) => j.relId === relId);
  if (nRelJobs === 1) {
    notifyGraphingStatusChanged(relId, true);
  } else if (nRelJobs === 0) {
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
    job.proc?.stdin?.write("\n");
    return;
  }

  const nBefore = countJobs();
  activeJobs = activeJobs.filter((j) => j !== job);
  sleepingJobs = sleepingJobs.filter((j) => j !== job);
  sleepingJobs.push(job);
  const nAfter = countJobs();
  assert(nBefore === nAfter);

  updateQueue();
}

function notifyExportImageStatusChanged(progress: ExportImageProgress) {
  mainWindow?.webContents.send(ipc.exportImageStatusChanged, progress);
}

function notifyGraphingStatusChanged(relId: string, processing: boolean) {
  mainWindow?.webContents.send(ipc.graphingStatusChanged, relId, processing);
}

function notifyTileReady(
  relId: string,
  tileId: string,
  incrementVersion: boolean
) {
  const tile = relationById.get(relId)?.tiles.get(tileId);
  if (tile === undefined) {
    return;
  }

  if (incrementVersion) {
    tile.version = (tile.version ?? 0) + 1;
  }

  mainWindow?.webContents.send(
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
      activeJobs.push(job);
      job.proc?.stdin?.write("\n");
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
      job.proc.stdin?.on("error", () => {
        // ignore
      });

      const nBefore = countJobs();
      if (activeJobs.length < MAX_ACTIVE_JOBS) {
        activeJobs.push(job);
        job.proc.stdin?.write("\n");
      } else {
        sleepingJobs.unshift(job);
      }
      const nAfter = countJobs();
      assert(nAfter === nBefore + 1);

      checkAndNotifyGraphingStatusChanged(job.relId);
    }
  }
}
