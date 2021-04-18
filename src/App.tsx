import "@fontsource/noto-sans/400.css";
import "./App.css";

import { initializeIcons } from "@fluentui/font-icons-mdl2";
import { Stack, ThemeProvider, useTheme } from "@fluentui/react";
import * as React from "react";
import { useRef } from "react";
import * as ReactDOM from "react-dom";
import { Provider } from "react-redux";
import { GraphBars } from "./GraphBars";
import { GraphCommandBar } from "./GraphCommandBar";
import { GraphView } from "./GraphView";
import { store } from "./models/store";

const App = () => {
  const graphViewRef = useRef<HTMLDivElement | null>(null);
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
        <GraphBars focusGraphView={focusGraphView} />
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
