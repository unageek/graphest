import {
  app,
  BrowserWindow,
  clipboard,
  dialog,
  Menu,
  shell,
  ipcMain as untypedIpcMain,
} from "electron";
import installExtension, {
  REACT_DEVELOPER_TOOLS,
  REDUX_DEVTOOLS,
} from "electron-devtools-installer";
import { autoUpdater } from "electron-updater";
import * as _ from "lodash";
import assert from "node:assert";
import {
  ChildProcess,
  execFile,
  ExecFileException,
  spawn,
} from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as url from "node:url";
import * as util from "node:util";
import { bignum } from "../common/bignumber";
import { Command } from "../common/command";
import {
  BASE_ZOOM_LEVEL,
  GRAPH_TILE_EXTENSION,
  GRAPH_TILE_SIZE,
  PERTURBATION_X,
  PERTURBATION_Y,
} from "../common/constants";
import { Document } from "../common/document";
import {
  ANTI_ALIASING_OPTIONS,
  EXPORT_GRAPH_TILE_SIZE,
  ExportImageProgress,
  MAX_EXPORT_IMAGE_SIZE,
  MAX_EXPORT_TIMEOUT,
} from "../common/exportImage";
import * as ipc from "../common/ipc";
import { SaveTo } from "../common/ipc";
import { Range } from "../common/range";
import * as result from "../common/result";
import { fromBase64Url, toBase64Url } from "./base64Url";
import { createMainMenu } from "./mainMenu";
import { deserialize, serialize } from "./serialize";

const fsPromises = fs.promises;

const ipcMain = untypedIpcMain as ipc.IpcMain;

interface TypedWebContents extends Electron.WebContents {
  send<T extends ipc.MessageToRenderer>(
    channel: T["channel"],
    ...args: T["args"]
  ): void;
}

interface BrowserWindowWithTypedWebContents extends Electron.BrowserWindow {
  webContents: TypedWebContents;
}

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

const URL_PREFIX = "graphest://";

interface GraphTask {
  args: string[];
}

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
  suffixArgs: string[];
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
const concatenateExec: string = getBundledExecutable("concatenate");
let currentPath: string | undefined;
let exportImageAbortController: AbortController | undefined;
const graphExec: string = getBundledExecutable("graph");
let lastSavedDoc: Document = {
  background: "white",
  center: [0, 0],
  foreground: "black",
  graphs: [],
  version: 1,
  zoomLevel: 6,
};
let mainMenu: Menu | undefined;
let mainWindow: BrowserWindowWithTypedWebContents | undefined;
let maybeUnsaved = false;
let nextExportImageId = 0;
let nextRelId = 0;
let postStartup: (() => void | Promise<void>) | undefined = () =>
  openUrl(
    "graphest://eyJjZW50ZXIiOlswLDBdLCJncmFwaHMiOlt7ImNvbG9yIjoicmdiYSgwLCA3OCwgMTQwLCAwLjgpIiwicGVuU2l6ZSI6MSwicmVsYXRpb24iOiJ5ID0gc2luKHgpIn1dLCJ2ZXJzaW9uIjoxLCJ6b29tTGV2ZWwiOjZ9"
  );
let postUnload: (() => void | Promise<void>) | undefined;
const relationById = new Map<string, Relation>();
const relKeyToRelId = new Map<string, string>();

function createMainWindow() {
  mainWindow = new BrowserWindow({
    height: 690,
    minHeight: 300,
    minWidth: 300,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      spellcheck: false,
    },
    width: 920,
  })
    .on("close", (e) => {
      if (maybeUnsaved) {
        postUnload = () => mainWindow?.close();
        mainWindow?.webContents.send<ipc.InitiateUnload>(ipc.initiateUnload);
        e.preventDefault();
      }
    })
    .on("closed", () => {
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

// https://www.electronjs.org/docs/latest/tutorial/launch-app-from-url-in-another-app
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient("graphest", process.execPath, [
      path.resolve(process.argv[1]),
    ]);
  }
} else {
  app.setAsDefaultProtocolClient("graphest");
}

app.whenReady().then(async () => {
  resetBrowserZoom();

  mainMenu = createMainMenu({
    [Command.AbortGraphing]: () => abortJobs(),
    [Command.HighResolution]: () => {
      mainWindow?.webContents.send<ipc.CommandInvoked>(
        ipc.commandInvoked,
        Command.HighResolution
      );
    },
    [Command.NewDocument]: () => newDocument(),
    [Command.Open]: () => open(),
    [Command.OpenFromClipboard]: () => openFromClipboard(),
    [Command.Save]: () => {
      mainWindow?.webContents.send<ipc.InitiateSave>(
        ipc.initiateSave,
        SaveTo.CurrentFile
      );
    },
    [Command.SaveAs]: () => {
      mainWindow?.webContents.send<ipc.InitiateSave>(
        ipc.initiateSave,
        SaveTo.NewFile
      );
    },
    [Command.SaveToClipboard]: () => {
      mainWindow?.webContents.send<ipc.InitiateSave>(
        ipc.initiateSave,
        SaveTo.Clipboard
      );
    },
    [Command.ShowAxes]: () => {
      mainWindow?.webContents.send<ipc.CommandInvoked>(
        ipc.commandInvoked,
        Command.ShowAxes
      );
    },
    [Command.ShowMajorGrid]: () => {
      mainWindow?.webContents.send<ipc.CommandInvoked>(
        ipc.commandInvoked,
        Command.ShowMajorGrid
      );
    },
    [Command.ShowMinorGrid]: () => {
      mainWindow?.webContents.send<ipc.CommandInvoked>(
        ipc.commandInvoked,
        Command.ShowMinorGrid
      );
    },
  });
  Menu.setApplicationMenu(mainMenu);

  try {
    // https://github.com/MarshallOfSound/electron-devtools-installer/issues/195#issuecomment-932634933
    await installExtension([REACT_DEVELOPER_TOOLS, REDUX_DEVTOOLS], {
      loadExtensionOptions: { allowFileAccess: true },
    });
  } catch {
    // Maybe no internet connection.
  }

  createMainWindow();
  autoUpdater.checkForUpdatesAndNotify();
});

app.on("open-file", (_, path) => {
  openFile(path);
});

app.on("open-url", (_, url) => {
  openUrl(url);
});

app.on("quit", () => {
  abortJobs();
  fs.rmSync(baseOutDir, { recursive: true });
});

app.on("window-all-closed", () => {
  app.quit();
});

ipcMain.handle<ipc.AbortExportImage>(ipc.abortExportImage, async () => {
  exportImageAbortController?.abort();
});

ipcMain.handle<ipc.AbortGraphing>(
  ipc.abortGraphing,
  async (_, relId, tileId) => {
    abortJobs(
      (j) => j.relId === relId && (tileId === undefined || j.tileId === tileId)
    );
  }
);

ipcMain.handle<ipc.ExportImage>(ipc.exportImage, async (__, data, opts) => {
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
      path &&
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

  const entries = [];
  for (const graph of data.graphs) {
    const rel = relationById.get(graph.relId);
    if (rel === undefined) {
      return;
    }

    entries.push({
      path: path.join(outDir, nextExportImageId.toString() + ".png"),
      rel,
      tilePathPrefix: path.join(outDir, nextExportImageId.toString() + "-"),
      tilePathSuffix: ".png",
      ...graph,
    });
    nextExportImageId++;
  }

  const abortController = new AbortController();
  exportImageAbortController = abortController;

  function runGraphTasks(tasks: GraphTask[]) {
    return new Promise((resolve, reject) => {
      const messages = new Set<string>();
      const maxProcesses = os.cpus().length;
      const totalTasks = tasks.length;
      let completedTasks = 0;

      notifyExportImageStatusChanged({
        messages: [...messages],
        numTiles: totalTasks,
        numTilesRendered: completedTasks,
      });

      async function run() {
        const task = tasks.shift();
        if (task === undefined) {
          return;
        }

        try {
          const { stderr } = await util.promisify(execFile)(
            graphExec,
            task.args,
            { signal: abortController.signal }
          );
          completedTasks++;

          if (stderr) {
            messages.add(stderr.trimEnd());
          }
          notifyExportImageStatusChanged({
            messages: [...messages],
            numTiles: totalTasks,
            numTilesRendered: completedTasks,
          });

          if (completedTasks === totalTasks) {
            resolve(null);
          } else {
            run();
          }
        } catch (e) {
          const { stderr } = e as ExecFileException;
          if (typeof stderr === "string" && stderr) {
            console.warn(stderr.trimEnd());
          }
          console.error("`graph` failed:", `'${task.args.join("' '")}'`);
          abortController.abort();
          reject();
        }
      }

      for (let i = 0; i < maxProcesses; i++) {
        run();
      }
    });
  }

  const xTiles = Math.ceil(
    (opts.antiAliasing * opts.width) / EXPORT_GRAPH_TILE_SIZE
  );
  const yTiles = Math.ceil(
    (opts.antiAliasing * opts.height) / EXPORT_GRAPH_TILE_SIZE
  );
  const tileWidth = opts.width / xTiles;
  const tileHeight = opts.height / yTiles;

  const pixelWidth = bounds[1].minus(bounds[0]).div(opts.width);
  const pixelHeight = bounds[3].minus(bounds[2]).div(opts.height);
  const x0 = bounds[0].minus(pixelWidth.times(PERTURBATION_X));
  const y1 = bounds[3].minus(pixelHeight.times(PERTURBATION_Y));

  let graphTasks: GraphTask[] = [];
  for (let k = 0; k < entries.length; k++) {
    const entry = entries[k];

    for (let iTile = 0; iTile < yTiles; iTile++) {
      const i0 = Math.round(iTile * tileHeight);
      const i1 = Math.round((iTile + 1) * tileHeight);
      const height = i1 - i0;
      for (let jTile = 0; jTile < xTiles; jTile++) {
        const j0 = Math.round(jTile * tileWidth);
        const j1 = Math.round((jTile + 1) * tileWidth);
        const width = j1 - j0;

        const bounds = [
          x0.plus(pixelWidth.times(j0)),
          x0.plus(pixelWidth.times(j1)),
          y1.minus(pixelHeight.times(i1)),
          y1.minus(pixelHeight.times(i0)),
        ];

        const path = `${entry.tilePathPrefix}${k}-${iTile}-${jTile}${entry.tilePathSuffix}`;
        const args = [
          "--bounds",
          ...bounds.map((b) => b.toString()),
          "--gray-alpha",
          "--output",
          path,
          "--output-once",
          "--pen-size",
          entry.penSize.toString(),
          "--size",
          width.toString(),
          height.toString(),
          "--ssaa",
          opts.antiAliasing.toString(),
          "--timeout",
          (1000 * opts.timeout).toString(),
          ...entry.rel.suffixArgs,
        ];
        graphTasks.push({ args });
      }
    }
  }

  // Try to make the progress uniform.
  graphTasks = _.shuffle(graphTasks);

  try {
    await runGraphTasks(graphTasks);
  } catch {
    return;
  }

  for (let k = 0; k < entries.length; k++) {
    const entry = entries[k];

    const args = [
      "--output",
      entry.path,
      "--prefix",
      `${entry.tilePathPrefix}${k}-`,
      "--size",
      opts.width.toString(),
      opts.height.toString(),
      "--suffix",
      entry.tilePathSuffix,
      "--x-tiles",
      xTiles.toString(),
      "--y-tiles",
      yTiles.toString(),
    ];
    try {
      const { stderr } = await util.promisify(execFile)(concatenateExec, args, {
        signal: abortController.signal,
      });
      if (stderr) {
        console.warn(stderr.trimEnd());
      }
    } catch (e) {
      const { stderr } = e as ExecFileException;
      if (typeof stderr === "string" && stderr) {
        console.warn(stderr.trimEnd());
      }
      console.error("`concatenate` failed:", `'${args.join("' '")}'`);
      return;
    }
  }

  const args = [
    ...entries.flatMap((entry) => ["--add", entry.path, entry.color]),
    "--background",
    opts.transparent ? "#00000000" : data.background,
    ...(opts.correctAlpha ? ["--correct-alpha"] : []),
    "--output",
    opts.path,
  ];
  try {
    const { stderr } = await util.promisify(execFile)(composeExec, args, {
      signal: abortController.signal,
    });
    if (stderr) {
      console.warn(stderr.trimEnd());
    }
  } catch (e) {
    const { stderr } = e as ExecFileException;
    if (typeof stderr === "string" && stderr) {
      console.warn(stderr.trimEnd());
    }
    console.error("`compose` failed:", `'${args.join("' '")}'`);
    return;
  }

  await shell.openPath(opts.path);
});

ipcMain.handle<ipc.GetDefaultExportImagePath>(
  ipc.getDefaultExportImagePath,
  async () => {
    try {
      return path.join(app.getPath("pictures"), "graph.png");
    } catch {
      return "";
    }
  }
);

ipcMain.handle<ipc.Ready>(ipc.ready, async () => {
  postStartup?.();
  postStartup = undefined;
});

ipcMain.handle<ipc.RequestRelation>(
  ipc.requestRelation,
  async (_, rel, graphId, highRes) => {
    try {
      let suffixArgs = [];
      // https://docs.microsoft.com/en-us/troubleshoot/windows-client/shell-experience/command-line-string-limitation
      if (rel.length > 5000) {
        const relationFile = path.join(baseOutDir, nextRelId.toString() + ".i");
        await fsPromises.writeFile(relationFile, rel);
        suffixArgs = ["--input", relationFile];
      } else {
        suffixArgs = ["--", rel];
      }
      const args = ["--dump-ast", "--parse", ...suffixArgs];
      const { stdout } = await util.promisify(execFile)(graphExec, args, {
        maxBuffer: 1024 * 1024 * 1024,
      });
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
          suffixArgs,
          tiles: new Map(),
        });
        relKeyToRelId.set(relKey, relId);
      }
      return result.ok(relId);
    } catch (e) {
      const { stderr } = e as ExecFileException;
      if (typeof stderr === "string") {
        const lines = stderr.split("\n");
        const start =
          parseInt((lines[1].match(/^.*:\d+:(\d+)/) as RegExpMatchArray)[1]) -
          1;
        const len = (lines[3].match(/~*$/) as RegExpMatchArray)[0].length;
        const message = (lines[1].match(/error: (.*)$/) as RegExpMatchArray)[1];
        return result.err({
          range: new Range(start, start + len),
          message,
        });
      } else {
        return result.err({
          range: new Range(0, 0),
          message: "unexpected error",
        });
      }
    }
  }
);

ipcMain.handle<ipc.RequestSave>(ipc.requestSave, async (_, doc, to) => {
  save(doc, to);
});

ipcMain.handle<ipc.RequestTile>(
  ipc.requestTile,
  async (_, relId, tileId, coords) => {
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
      const widthPerTile = bignum(
        GRAPH_TILE_SIZE * 2 ** (BASE_ZOOM_LEVEL - coords.z)
      );
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
          ...rel.suffixArgs,
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

ipcMain.handle<ipc.RequestUnload>(ipc.requestUnload, async (_, doc) => {
  unload(doc);
});

ipcMain.handle<ipc.ShowSaveDialog>(ipc.showSaveDialog, async (_, path) => {
  if (!mainWindow) return undefined;
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: path,
    filters: [{ name: "PNG", extensions: ["png"] }],
  });
  return result.filePath;
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

function defaultSavePath(): string {
  try {
    return path.join(app.getPath("documents"), "Untitled.graphest");
  } catch {
    return "";
  }
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

function getCurrentFilenameForDisplay(): string {
  return getFilenameForDisplay(currentPath);
}

function getFilenameForDisplay(thePath?: string): string {
  if (thePath !== undefined) {
    return path.basename(thePath, ".graphest");
  } else {
    return "Untitled";
  }
}

function newDocument() {
  openUrl(
    "graphest://eyJncmFwaHMiOlt7ImNvbG9yIjoicmdiYSgwLCA3OCwgMTQwLCAwLjgpIiwicGVuU2l6ZSI6MSwicmVsYXRpb24iOiIifV0sInZlcnNpb24iOjF9"
  );
}

function notifyExportImageStatusChanged(progress: ExportImageProgress) {
  mainWindow?.webContents.send<ipc.ExportImageStatusChanged>(
    ipc.exportImageStatusChanged,
    progress
  );
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
  const tile = relationById.get(relId)?.tiles.get(tileId);
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

async function open() {
  if (!mainWindow) return;

  const result = await dialog.showOpenDialog(mainWindow, {
    defaultPath: currentPath ?? defaultSavePath(),
    filters: [{ name: "Graphest Document", extensions: ["graphest"] }],
  });
  if (result.filePaths.length !== 1) return;
  const path = result.filePaths[0];

  openFile(path);
}

async function openFile(path: string) {
  if (!mainWindow) {
    postStartup = () => openFile(path);
    return;
  }

  if (maybeUnsaved) {
    postUnload = () => openFile(path);
    mainWindow.webContents.send<ipc.InitiateUnload>(ipc.initiateUnload);
    return;
  }

  try {
    const data = await fsPromises.readFile(path, { encoding: "utf8" });
    const doc = deserialize(data);
    currentPath = path;
    lastSavedDoc = doc;
    maybeUnsaved = true;
    mainWindow.setRepresentedFilename(path);
    mainWindow.setTitle(getCurrentFilenameForDisplay());
    mainWindow.webContents.send<ipc.Load>(ipc.load, doc);
  } catch (e) {
    console.log("open failed", e);
    dialog.showMessageBox(mainWindow, {
      message:
        `The document “${getFilenameForDisplay(path)}”` +
        " could not be opened.",
      type: "warning",
    });
  }
}

function openFromClipboard() {
  openUrl(clipboard.readText());
}

function openUrl(url: string) {
  if (!mainWindow) {
    postStartup = () => openUrl(url);
    return;
  }

  if (maybeUnsaved) {
    postUnload = () => openUrl(url);
    mainWindow.webContents.send<ipc.InitiateUnload>(ipc.initiateUnload);
    return;
  }

  if (url.startsWith(URL_PREFIX)) {
    try {
      const data = fromBase64Url(url.substring(URL_PREFIX.length));
      const doc = deserialize(data);
      currentPath = undefined;
      lastSavedDoc = doc;
      maybeUnsaved = true;
      mainWindow.setRepresentedFilename("");
      mainWindow.setTitle(getCurrentFilenameForDisplay());
      mainWindow.webContents.send<ipc.Load>(ipc.load, doc);
    } catch (e) {
      console.log("open failed", e);
    }
  }
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

async function save(doc: Document, to: SaveTo): Promise<boolean> {
  if (!mainWindow) return false;

  const data = serialize(doc);

  if (to === SaveTo.Clipboard) {
    clipboard.writeText(URL_PREFIX + toBase64Url(data));
    return true;
  }

  let path = currentPath;
  if (to === SaveTo.NewFile || path === undefined) {
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: path ?? defaultSavePath(),
      filters: [{ name: "Graphest Document", extensions: ["graphest"] }],
    });
    if (!result.filePath) return false;
    path = result.filePath;
  }

  try {
    await fsPromises.writeFile(path, data);
    if (process.platform === "darwin") {
      // Hide filename extension.
      await spawn("SetFile", ["-a", "E", path]);
    }
    currentPath = path;
    lastSavedDoc = doc;
    maybeUnsaved = false;
    mainWindow.setRepresentedFilename(path);
    mainWindow.setTitle(getCurrentFilenameForDisplay());
    return true;
  } catch (e) {
    console.log("save failed", e);
    await dialog.showMessageBox({
      type: "warning",
      message:
        `The document “${getCurrentFilenameForDisplay()}” could not be saved` +
        (to === SaveTo.CurrentFile
          ? "."
          : ` as “${getFilenameForDisplay(path)}”.`),
    });
    return false;
  }
}

async function unload(doc: Document) {
  if (_.isEqual(doc, lastSavedDoc)) {
    maybeUnsaved = false;
    postUnload?.();
    postUnload = undefined;
    return;
  }

  const result = await dialog.showMessageBox({
    message: `Do you want to save the changes made to the document “${getCurrentFilenameForDisplay()}”?`,
    detail: "Your changes will be lost if you don't save them.",
    buttons: ["Save…", "Don't Save", "Cancel"],
    noLink: true,
  });
  if (
    (result.response === 0 && (await save(doc, SaveTo.CurrentFile))) ||
    result.response === 1
  ) {
    maybeUnsaved = false;
    postUnload?.();
    postUnload = undefined;
  }
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
