import { initializeIcons } from "@fluentui/font-icons-mdl2";
import { Stack, ThemeProvider, useTheme } from "@fluentui/react";
import "@fontsource/dejavu-mono/400.css";
import "@fontsource/noto-sans/400.css";
import * as Color from "color";
import * as React from "react";
import { useEffect, useRef } from "react";
import * as ReactDOM from "react-dom";
import { Provider, useDispatch } from "react-redux";
import { Command } from "../common/command";
import { ExportImageOptions } from "../common/exportImage";
import * as ipc from "../common/ipc";
import { RequestRelationResult } from "../common/ipc";
import "./App.css";
import { ExportImageDialog } from "./ExportImageDialog";
import { GraphBars } from "./GraphBars";
import { GraphCommandBar } from "./GraphCommandBar";
import { GraphView } from "./GraphView";
import {
  setExportImageProgress,
  setHighRes,
  setLastExportImageOpts,
  setShowAxes,
  setShowExportDialog,
  setShowMajorGrid,
  setShowMinorGrid,
  useSelector,
} from "./models/app";
import { store } from "./models/store";

const abortExportImage = async () => {
  await window.ipcRenderer.invoke<ipc.AbortExportImage>(ipc.abortExportImage);
};

const exportImage = async (opts: ExportImageOptions) => {
  const state = store.getState();
  const entries = [];

  for (const graphId of state.graphs.allIds) {
    const graph = state.graphs.byId[graphId];
    const { color, relId } = graph;
    entries.push({ color: new Color(color).hexa(), relId });
  }

  store.dispatch(
    setExportImageProgress({
      lastStderr: "",
      // 1x1 transparent image.
      lastUrl:
        "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
      progress: 0,
    })
  );

  await window.ipcRenderer.invoke<ipc.ExportImage>(
    ipc.exportImage,
    entries,
    opts
  );
};

const openSaveDialog = async (path: string): Promise<string | undefined> => {
  return await window.ipcRenderer.invoke<ipc.OpenSaveDialog>(
    ipc.openSaveDialog,
    path
  );
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

const App = () => {
  const dispatch = useDispatch();
  const exportImageOpts = useSelector((s) => s.lastExportImageOpts);
  const graphViewRef = useRef<HTMLDivElement>(null);
  const showExportDialog = useSelector((s) => s.showExportDialog);
  const theme = useTheme();

  function focusGraphView() {
    graphViewRef.current?.focus();
  }

  useEffect(() => {
    (async function load() {
      const path = await window.ipcRenderer.invoke<ipc.GetDefaultImageFilePath>(
        ipc.getDefaultImageFilePath
      );
      store.dispatch(
        setLastExportImageOpts({
          ...exportImageOpts,
          path,
        })
      );
    })();
  }, []);

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
      {showExportDialog && (
        <ExportImageDialog
          abort={abortExportImage}
          dismiss={() => dispatch(setShowExportDialog(false))}
          exportImage={exportImage}
          openSaveDialog={openSaveDialog}
          opts={exportImageOpts}
          saveOpts={(opts) => {
            dispatch(setLastExportImageOpts(opts));
          }}
        />
      )}
    </>
  );
};

initializeIcons();

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
    case Command.Export:
      store.dispatch(setShowExportDialog(true));
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
