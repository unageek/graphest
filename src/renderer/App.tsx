import {
  FluentProvider,
  tokens,
  webDarkTheme,
  webLightTheme,
} from "@fluentui/react-components";
import "@fontsource/dejavu-mono/400.css";
import "@fontsource/noto-sans/400.css";
import Color from "color";
import { StrictMode, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Provider, useDispatch } from "react-redux";
import { Command } from "../common/command";
import { Document } from "../common/document";
import { ExportImageGraph, ExportImageOptions } from "../common/exportImage";
import * as ipc from "../common/ipc";
import { RequestRelationResult } from "../common/ipc";
import "./App.css";
import { ColorsDialog } from "./ColorsDialog";
import { CommandBar } from "./CommandBar";
import { GoToDialog } from "./GoToDialog";
import { GraphBars } from "./GraphBars";
import { GraphView } from "./GraphView";
import {
  addGraph,
  removeAllGraphs,
  setCenter,
  setExportImageProgress,
  setGraphBackground,
  setGraphForeground,
  setHighRes,
  setLastExportImageOpts,
  setResetView,
  setShowAxes,
  setShowColorsDialog,
  setShowExportImageDialog,
  setShowGoToDialog,
  setShowMajorGrid,
  setShowMinorGrid,
  setTheme,
  setZoomLevel,
  useSelector,
} from "./models/app";
import { store } from "./models/store";
import { RenderDialog } from "./RenderDialog";

const abortExportImage = async () => {
  await window.ipcRenderer.invoke<ipc.AbortExportImage>(ipc.abortExportImage);
};

const exportImage = async (opts: ExportImageOptions) => {
  const state = store.getState();
  const graphs: ExportImageGraph[] = [];

  for (const graphId of state.graphs.allIds) {
    const graph = state.graphs.byId[graphId];
    const { color, penSize, relId } = graph;
    graphs.push({ color: new Color(color).hexa(), penSize, relId });
  }

  store.dispatch(
    setExportImageProgress({
      messages: [],
      numTiles: 0,
      numTilesRendered: 0,
    }),
  );

  await window.ipcRenderer.invoke<ipc.ExportImage>(
    ipc.exportImage,
    { background: new Color(state.graphBackground).hexa(), graphs },
    opts,
  );
};

const getDocument = (): Document => {
  const s = store.getState();
  return {
    background: s.graphBackground,
    center: s.center,
    foreground: s.graphForeground,
    graphs: s.graphs.allIds.map((id) => {
      const g = s.graphs.byId[id];
      return {
        color: g.color,
        penSize: g.penSize,
        relation: g.relation,
      };
    }),
    version: 1,
    zoomLevel: s.zoomLevel,
  };
};

const requestRelation = async (
  rel: string,
  graphId: string,
  highRes: boolean,
): Promise<RequestRelationResult> => {
  return await window.ipcRenderer.invoke<ipc.RequestRelation>(
    ipc.requestRelation,
    rel,
    graphId,
    highRes,
  );
};

const showSaveDialog = async (path: string): Promise<string | undefined> => {
  return await window.ipcRenderer.invoke<ipc.ShowSaveDialog>(
    ipc.showSaveDialog,
    path,
  );
};

const App = () => {
  const dispatch = useDispatch();
  const center = useSelector((s) => s.center);
  const exportImageOpts = useSelector((s) => s.lastExportImageOpts);
  const graphViewRef = useRef<HTMLDivElement>(null);
  const showColorsDialog = useSelector((s) => s.showColorsDialog);
  const showExportImageDialog = useSelector((s) => s.showExportImageDialog);
  const showGoToDialog = useSelector((s) => s.showGoToDialog);
  const appTheme = useSelector((s) => s.theme);
  const zoomLevel = useSelector((s) => s.zoomLevel);

  function focusGraphView() {
    graphViewRef.current?.focus();
  }

  useEffect(() => {
    (async function () {
      const path =
        await window.ipcRenderer.invoke<ipc.GetDefaultExportImagePath>(
          ipc.getDefaultExportImagePath,
        );
      store.dispatch(
        setLastExportImageOpts({
          ...exportImageOpts,
          path,
        }),
      );
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <FluentProvider
      style={{ height: "100%" }}
      theme={appTheme === "light" ? webLightTheme : webDarkTheme}
    >
      <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <div
          style={{
            boxShadow: tokens.shadow4,
            zIndex: 2000, // To show on top of the <GraphView>.
          }}
        >
          <GraphBars
            focusGraphView={focusGraphView}
            requestRelation={requestRelation}
          />
          <CommandBar />
        </div>
        <GraphView grow ref={graphViewRef} />
      </div>
      {showColorsDialog && (
        <ColorsDialog dismiss={() => dispatch(setShowColorsDialog(false))} />
      )}
      {showExportImageDialog && (
        <RenderDialog
          abort={abortExportImage}
          dismiss={() => dispatch(setShowExportImageDialog(false))}
          exportImage={exportImage}
          opts={exportImageOpts}
          saveOpts={(opts) => {
            dispatch(setLastExportImageOpts(opts));
          }}
          showSaveDialog={showSaveDialog}
        />
      )}
      {showGoToDialog && (
        <GoToDialog
          center={center}
          dismiss={() => dispatch(setShowGoToDialog(false))}
          goTo={(center, zoomLevel) => {
            store.dispatch(setCenter(center));
            store.dispatch(setZoomLevel(zoomLevel));
            store.dispatch(setResetView(true));
          }}
          zoomLevel={zoomLevel}
        />
      )}
    </FluentProvider>
  );
};

createRoot(document.getElementById("app") as HTMLElement).render(
  <StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </StrictMode>,
);

window.ipcRenderer.on<ipc.CommandInvoked>(ipc.commandInvoked, (_, item) => {
  const state = store.getState();
  switch (item) {
    case Command.HighResolution:
      store.dispatch(setHighRes(!state.highRes));
      break;
    case Command.ShowAxes:
      store.dispatch(setShowAxes(!state.showAxes));
      break;
    case Command.ShowMajorGrid:
      store.dispatch(setShowMajorGrid(!state.showMajorGrid));
      break;
    case Command.ShowMinorGrid:
      store.dispatch(setShowMinorGrid(!state.showMinorGrid));
      break;
  }
});

window.ipcRenderer.on<ipc.ExportImageStatusChanged>(
  ipc.exportImageStatusChanged,
  (_, progress) => {
    store.dispatch(setExportImageProgress(progress));
  },
);

window.ipcRenderer.on<ipc.InitiateSave>(ipc.initiateSave, (_, to) => {
  window.ipcRenderer.invoke<ipc.RequestSave>(
    ipc.requestSave,
    getDocument(),
    to,
  );
});

window.ipcRenderer.on<ipc.InitiateUnload>(ipc.initiateUnload, () => {
  window.ipcRenderer.invoke<ipc.RequestUnload>(
    ipc.requestUnload,
    getDocument(),
  );
});

window.ipcRenderer.on<ipc.Load>(ipc.load, (_, state) => {
  store.dispatch(removeAllGraphs());
  store.dispatch(setCenter(state.center as [number, number]));
  store.dispatch(setZoomLevel(state.zoomLevel));
  store.dispatch(setResetView(true));
  store.dispatch(setGraphBackground(state.background));
  store.dispatch(setGraphForeground(state.foreground));
  for (const g of state.graphs) {
    store.dispatch(addGraph(g));
  }
});

window.ipcRenderer.invoke<ipc.Ready>(ipc.ready);

const colorThemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

store.dispatch(setTheme(colorThemeQuery.matches ? "dark" : "light"));

colorThemeQuery.addEventListener("change", (e) => {
  store.dispatch(setTheme(e.matches ? "dark" : "light"));
});
