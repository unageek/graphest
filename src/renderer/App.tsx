import { initializeIcons } from "@fluentui/font-icons-mdl2";
import { Stack, ThemeProvider, useTheme } from "@fluentui/react";
import "@fontsource/dejavu-mono/400.css";
import "@fontsource/noto-sans/400.css";
import * as React from "react";
import { useRef } from "react";
import * as ReactDOM from "react-dom";
import { Provider } from "react-redux";
import { Command } from "../common/command";
import * as ipc from "../common/ipc";
import { RequestRelationResult } from "../common/ipc";
import "./App.css";
import { GraphBars } from "./GraphBars";
import { GraphCommandBar } from "./GraphCommandBar";
import { GraphView } from "./GraphView";
import { setHighRes, setShowAxes, setShowGrid } from "./models/app";
import { store } from "./models/store";

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
  const graphViewRef = useRef<HTMLDivElement>(null);
  const theme = useTheme();

  function focusGraphView() {
    graphViewRef.current?.focus();
  }

  return (
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
    case Command.HighResolution:
      store.dispatch(setHighRes(!state.highRes));
      break;
    case Command.ShowAxes:
      store.dispatch(setShowAxes(!state.showAxes));
      break;
    case Command.ShowGrid:
      store.dispatch(setShowGrid(!state.showGrid));
      break;
  }
});
