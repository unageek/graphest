import { registerIcons, Stack, ThemeProvider, useTheme } from "@fluentui/react";
import {
  AddIcon,
  CancelIcon,
  CheckMarkIcon,
  ChevronDownIcon,
  ChevronDownSmallIcon,
  ChevronUpSmallIcon,
  DeleteIcon,
  ForwardIcon,
  GripperDotsVerticalIcon,
  HomeSolidIcon,
  MoreIcon,
  VariableIcon,
} from "@fluentui/react-icons-mdl2";
import "@fontsource/dejavu-mono/400.css";
import "@fontsource/noto-sans/400.css";
import "@fortawesome/fontawesome-free/js/fontawesome";
import "@fortawesome/fontawesome-free/js/solid";
import * as Color from "color";
import * as React from "react";
import { useEffect, useRef } from "react";
import * as ReactDOM from "react-dom";
import { Provider, useDispatch } from "react-redux";
import { Command } from "../common/command";
import { Document } from "../common/document";
import { ExportImageEntry, ExportImageOptions } from "../common/exportImage";
import * as ipc from "../common/ipc";
import { RequestRelationResult } from "../common/ipc";
import "./App.css";
import { ExportImageDialog } from "./ExportImageDialog";
import { GoToDialog } from "./GoToDialog";
import { GraphBars } from "./GraphBars";
import { GraphCommandBar } from "./GraphCommandBar";
import { GraphView } from "./GraphView";
import {
  addGraph,
  removeAllGraphs,
  setCenter,
  setExportImageProgress,
  setHighRes,
  setLastExportImageOpts,
  setResetView,
  setShowAxes,
  setShowExportImageDialog,
  setShowGoToDialog,
  setShowMajorGrid,
  setShowMinorGrid,
  setZoomLevel,
  useSelector,
} from "./models/app";
import { store } from "./models/store";

const abortExportImage = async () => {
  await window.ipcRenderer.invoke<ipc.AbortExportImage>(ipc.abortExportImage);
};

const exportImage = async (opts: ExportImageOptions) => {
  const state = store.getState();
  const entries: ExportImageEntry[] = [];

  for (const graphId of state.graphs.allIds) {
    const graph = state.graphs.byId[graphId];
    const { color, penSize, relId } = graph;
    entries.push({ color: new Color(color).hexa(), penSize, relId });
  }

  store.dispatch(
    setExportImageProgress({
      messages: [],
      progress: 0,
    })
  );

  await window.ipcRenderer.invoke<ipc.ExportImage>(
    ipc.exportImage,
    entries,
    opts
  );
};

const getDocument = (): Document => {
  const s = store.getState();
  return {
    center: s.center,
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
  highRes: boolean
): Promise<RequestRelationResult> => {
  return await window.ipcRenderer.invoke<ipc.RequestRelation>(
    ipc.requestRelation,
    rel,
    graphId,
    highRes
  );
};

const showSaveDialog = async (path: string): Promise<string | undefined> => {
  return await window.ipcRenderer.invoke<ipc.ShowSaveDialog>(
    ipc.showSaveDialog,
    path
  );
};

const App = () => {
  const dispatch = useDispatch();
  const center = useSelector((s) => s.center);
  const exportImageOpts = useSelector((s) => s.lastExportImageOpts);
  const graphViewRef = useRef<HTMLDivElement>(null);
  const showExportImageDialog = useSelector((s) => s.showExportImageDialog);
  const showGoToDialog = useSelector((s) => s.showGoToDialog);
  const theme = useTheme();
  const zoomLevel = useSelector((s) => s.zoomLevel);

  function focusGraphView() {
    graphViewRef.current?.focus();
  }

  useEffect(() => {
    (async function () {
      const path =
        await window.ipcRenderer.invoke<ipc.GetDefaultExportImagePath>(
          ipc.getDefaultExportImagePath
        );
      store.dispatch(
        setLastExportImageOpts({
          ...exportImageOpts,
          path,
        })
      );
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <Stack verticalFill>
        <Stack
          styles={{
            root: {
              boxShadow: theme.effects.elevation4,
              zIndex: 2000, // To show on top of the <GraphView>.
            },
          }}
        >
          <GraphBars
            focusGraphView={focusGraphView}
            requestRelation={requestRelation}
          />
          <GraphCommandBar />
        </Stack>
        <GraphView grow ref={graphViewRef} />
      </Stack>
      {showExportImageDialog && (
        <ExportImageDialog
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
    </>
  );
};

const ICON_CLASS_NAME = "fluent-ui-icon";
registerIcons({
  icons: {
    Add: <AddIcon className={ICON_CLASS_NAME} />,
    Cancel: <CancelIcon className={ICON_CLASS_NAME} />,
    CheckMark: <CheckMarkIcon className={ICON_CLASS_NAME} />,
    ChevronDown: <ChevronDownIcon className={ICON_CLASS_NAME} />,
    ChevronDownSmall: <ChevronDownSmallIcon className={ICON_CLASS_NAME} />,
    ChevronUpSmall: <ChevronUpSmallIcon className={ICON_CLASS_NAME} />,
    Delete: <DeleteIcon className={ICON_CLASS_NAME} />,
    Forward: <ForwardIcon className={ICON_CLASS_NAME} />,
    GripperDotsVertical: (
      <GripperDotsVerticalIcon className={ICON_CLASS_NAME} />
    ),
    HomeSolid: <HomeSolidIcon className={ICON_CLASS_NAME} />,
    More: <MoreIcon className={ICON_CLASS_NAME} />,
    Variable: <VariableIcon className={ICON_CLASS_NAME} />,
  },
});

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <ThemeProvider style={{ height: "100%" }}>
        <App />
      </ThemeProvider>
    </Provider>
  </React.StrictMode>,
  document.getElementById("app")
);

window.ipcRenderer.on<ipc.CommandInvoked>(ipc.commandInvoked, (_, item) => {
  const state = store.getState();
  switch (item) {
    case Command.ExportImage:
      store.dispatch(setShowExportImageDialog(true));
      break;
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
  }
);

window.ipcRenderer.on<ipc.InitiateSave>(ipc.initiateSave, (_, to) => {
  window.ipcRenderer.invoke<ipc.RequestSave>(
    ipc.requestSave,
    getDocument(),
    to
  );
});

window.ipcRenderer.on<ipc.InitiateUnload>(ipc.initiateUnload, () => {
  window.ipcRenderer.invoke<ipc.RequestUnload>(
    ipc.requestUnload,
    getDocument()
  );
});

window.ipcRenderer.on<ipc.Load>(ipc.load, (_, state) => {
  store.dispatch(removeAllGraphs());
  store.dispatch(setCenter(state.center as [number, number]));
  store.dispatch(setZoomLevel(state.zoomLevel));
  store.dispatch(setResetView(true));
  for (const g of state.graphs) {
    store.dispatch(addGraph(g));
  }
});

window.ipcRenderer.invoke<ipc.Ready>(ipc.ready);
