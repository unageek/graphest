import * as assert from "assert";
import { ChildProcess, execFile } from "child_process";
import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  MenuItemConstructorOptions,
  screen,
  shell,
} from "electron";
import installExtension, {
  REACT_DEVELOPER_TOOLS,
  REDUX_DEVTOOLS,
} from "electron-devtools-installer";
import { autoUpdater } from "electron-updater";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { pathToFileURL } from "url";
import * as util from "util";
import { bignum } from "../common/BigNumber";
import {
  BASE_ZOOM_LEVEL,
  EXTENDED_GRAPH_TILE_SIZE,
  GRAPH_TILE_EXTENSION,
  GRAPH_TILE_SIZE,
} from "../common/constants";
import * as ipc from "../common/ipc";
import { MenuItem } from "../common/MenuItem";
import { Range } from "../common/range";
import * as result from "../common/result";

const fsPromises = fs.promises;

// Lifecycle of a job
//
//        +----------+        +----------+        +----------+
//   ●--->|  Queued  +------->|  Active  +------->|  Exited  |
//        +----------+        +--+----+--+        +----------+
//                               | Λ  |
//                               | |  +--------------+
//                      Pause *1 | | Resume *2       | SIGKILL
//                               V |                 V
//                            +----+-----+        +----------+
//                            | Sleeping +------->| Aborted  |
//                            +----------+  SIG-  +----------+
//                                          KILL
//
// *1. `graph` pauses execution right after initialization and every time it finishes writing the output image.
// *2. `graph` resumes execution when a newline character is written to its stdin.

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

let queuedJobs: Job[] = [];
let activeJobs: Job[] = [];
let sleepingJobs: Job[] = [];

const baseOutDir: string = fs.mkdtempSync(path.join(os.tmpdir(), "graphest-"));
const graphExec: string = path.join(
  __dirname,
  // ".exe" is required for pointing to executables inside .asar archives.
  process.platform === "win32" ? "graph.exe" : "graph"
);
let mainMenu: Menu | undefined;
let mainWindow: BrowserWindow | undefined;
let nextRelId = 0;
const relationById = new Map<string, Relation>();
const astToRelationId = new Map<string, string>();
const astToRelationIdHighRes = new Map<string, string>();

function createMainMenu(): Menu {
  // https://www.electronjs.org/docs/api/menu#examples
  // https://github.com/electron/electron/blob/main/lib/browser/api/menu-item-roles.ts
  const isMac = process.platform === "darwin";
  const isRetina = screen.getPrimaryDisplay().scaleFactor === 2;
  return Menu.buildFromTemplate([
    ...(isMac ? [{ role: "appMenu" }] : []),
    {
      role: "fileMenu",
      label: "&File",
      submenu: [
        // The Close menu is required for closing the about panel.
        { role: "close" },
        { type: "separator" },
        ...(isMac ? [] : [{ role: "quit" }]),
      ],
    },
    { role: "editMenu", label: "&Edit" },
    {
      label: "&Graph",
      submenu: [
        {
          id: MenuItem.ShowAxes,
          label: "Show &Axes",
          accelerator: "Alt+CmdOrCtrl+A",
          type: "checkbox",
          checked: true,
          click: () => {
            mainWindow?.webContents.send(
              ipc.menuItemInvoked,
              MenuItem.ShowAxes
            );
          },
        },
        {
          id: MenuItem.ShowGrid,
          label: "Show &Grid",
          accelerator: "Alt+CmdOrCtrl+G",
          type: "checkbox",
          checked: true,
          click: () => {
            mainWindow?.webContents.send(
              ipc.menuItemInvoked,
              MenuItem.ShowGrid
            );
          },
        },
        {
          type: "separator",
        },
        ...(isRetina
          ? [
              {
                id: MenuItem.HighResolution,
                label: "&High Resolution",
                type: "checkbox",
                click: () => {
                  mainWindow?.webContents.send(
                    ipc.menuItemInvoked,
                    MenuItem.HighResolution
                  );
                },
              },
            ]
          : []),
        {
          type: "separator",
        },
        {
          id: MenuItem.AbortGraphing,
          label: "A&bort Graphing",
          accelerator: "Esc",
          click: () => {
            abortJobs();
          },
        },
      ],
    },
    {
      role: "windowMenu",
      label: "&Window",
      submenu: [
        ...(isMac ? [{ role: "minimize" }, { role: "zoom" }] : []),
        // On macOS, it seems common to place the Toggle Full Screen menu under the Window menu
        // if there is nothing else to be placed under the View menu.
        { role: "togglefullscreen", label: "Toggle &Full Screen" },
        { type: "separator" },
        ...(isMac ? [{ role: "front" }] : []),
      ],
    },
    {
      role: "help",
      label: "&Help",
      submenu: [
        {
          label: "Graphest &Help",
          accelerator: isMac ? "" : "F1",
          click: async () => {
            await shell.openExternal(
              "https://unageek.github.io/graphest/guide/"
            );
          },
        },
        { type: "separator" },
        {
          label: "What's New",
          click: async () => {
            await shell.openExternal(
              "https://github.com/unageek/graphest/releases"
            );
          },
        },
        {
          label: "Example Relations",
          click: async () => {
            await shell.openExternal(
              "https://github.com/unageek/graphest/blob/main/Examples.md"
            );
          },
        },
        { type: "separator" },
        { role: "toggleDevTools" },
        { type: "separator" },
        ...(isMac ? [] : [{ role: "about" }]),
      ],
    },
  ] as MenuItemConstructorOptions[]);
}

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
  mainMenu = createMainMenu();
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

ipcMain.handle(ipc.abortGraphing, async (_, relId: string, tileId?: string) => {
  abortJobs((j) => j.relId === relId && j.tileId === tileId);
});

ipcMain.handle(
  ipc.requestRelation,
  async (
    _,
    rel: string,
    highRes: boolean
  ): Promise<ipc.RequestRelationResult> => {
    try {
      const args = ["--parse", "--dump-ast", "--", rel];
      const { stdout } = await util.promisify(execFile)(graphExec, args);
      const ast = stdout.split("\n")[0];
      let relId = highRes
        ? astToRelationIdHighRes.get(ast)
        : astToRelationId.get(ast);
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
        (highRes ? astToRelationIdHighRes : astToRelationId).set(ast, relId);
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
      // which could make lines such as `y = x` look thicker.
      const pixelOffsetX = bignum(
        (0.5 + 1.2345678901234567e-3) / (retinaScale * GRAPH_TILE_SIZE)
      );
      const pixelOffsetY = bignum(
        (0.5 + 1.3456789012345678e-3) / (retinaScale * GRAPH_TILE_SIZE)
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
        url: pathToFileURL(outFile).href,
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
          (retinaScale * EXTENDED_GRAPH_TILE_SIZE).toString(),
          (retinaScale * EXTENDED_GRAPH_TILE_SIZE).toString(),
          "--padding-right",
          (retinaScale * GRAPH_TILE_EXTENSION).toString(),
          "--padding-bottom",
          (retinaScale * GRAPH_TILE_EXTENSION).toString(),
          "--dilate",
          retinaScale === 2 ? "1,1,0;1,1,0;0,0,0" : "1",
          "--gray-alpha",
          "--output",
          outFile,
          "--mem-limit",
          JOB_MEM_LIMIT.toString(),
          "--pause-per-output",
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
  const abortGraphingMenu = mainMenu?.getMenuItemById(MenuItem.AbortGraphing);
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
