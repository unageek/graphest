import {
  FluentProvider,
  Input,
  Label,
  makeStyles,
  Subtitle2,
  useId,
} from "@fluentui/react-components";
import { clamp } from "lodash";
import { ReactNode, StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import * as ipc from "../common/ipc";
import { tryParseInteger } from "../common/parse";
import {
  getDefaultRenderingParallelism,
  getMaxRenderingParallelism,
  Settings,
} from "../common/settings";
import "./App.css";
import { useSystemTheme } from "./styles";

interface IntegerInputProps {
  id: string;
  max: number;
  onChange: (value: number) => void;
  unit: string;
  value: number;
}

// Applies an integer as soon as it is typed in, clamped to the range.
// Leaving the input shows the value in effect, discarding anything invalid.
const IntegerInput = (props: IntegerInputProps): ReactNode => {
  const [text, setText] = useState(props.value.toString());
  const styles = useStyles();

  return (
    <Input
      className={styles.input}
      contentAfter={props.unit}
      id={props.id}
      onBlur={() => setText(props.value.toString())}
      onChange={(_, { value }) => {
        setText(value);
        const result = tryParseInteger(value);
        if (result.ok !== undefined) {
          props.onChange(clamp(result.ok, 1, props.max));
        }
      }}
      value={text}
    />
  );
};

interface SettingsWindowProps {
  initialSettings: Settings;
  systemInformation: ipc.SystemInformation;
}

const SettingsWindow = (props: SettingsWindowProps): ReactNode => {
  const cpuLimitId = useId();
  const graphingId = useId();
  const memoryLimitId = useId();
  const parallelismId = useId();
  const renderingId = useId();
  const [settings, setSettings] = useState(props.initialSettings);
  const styles = useStyles();
  const theme = useSystemTheme();

  const update = (newSettings: Settings) => {
    setSettings(newSettings);
    window.ipcRenderer.invoke<ipc.SetSettings>(ipc.setSettings, newSettings);
  };

  const renderingParallelism =
    settings.renderingParallelism ??
    getDefaultRenderingParallelism(props.systemInformation);

  return (
    <FluentProvider className={styles.root} theme={theme}>
      <div aria-labelledby={graphingId} className={styles.group} role="group">
        <Subtitle2 as="h2" className={styles.heading} id={graphingId}>
          Graphing
        </Subtitle2>

        <Label className={styles.label} htmlFor={cpuLimitId}>
          CPU limit:
        </Label>
        <IntegerInput
          id={cpuLimitId}
          max={props.systemInformation.logicalCores}
          onChange={(value) => {
            update({ ...settings, cpuLimit: value });
          }}
          unit={settings.cpuLimit > 1 ? "cores" : "core"}
          value={settings.cpuLimit}
        />

        <Label className={styles.label} htmlFor={memoryLimitId}>
          Memory limit:
        </Label>
        <IntegerInput
          id={memoryLimitId}
          max={props.systemInformation.memory}
          onChange={(value) => {
            update({ ...settings, memoryLimit: value });
          }}
          unit="GB"
          value={settings.memoryLimit}
        />
      </div>

      <div aria-labelledby={renderingId} className={styles.group} role="group">
        <Subtitle2 as="h2" className={styles.heading} id={renderingId}>
          Rendering
        </Subtitle2>

        <Label className={styles.label} htmlFor={parallelismId}>
          Process in parallel:
        </Label>
        <IntegerInput
          id={parallelismId}
          max={getMaxRenderingParallelism(props.systemInformation)}
          onChange={(value) => {
            update({ ...settings, renderingParallelism: value });
          }}
          unit={renderingParallelism > 1 ? "tiles" : "tile"}
          value={renderingParallelism}
        />
      </div>
    </FluentProvider>
  );
};

const useStyles = makeStyles({
  // The groups share the columns, so that the inputs line up.
  root: {
    alignContent: "center",
    boxSizing: "border-box",
    columnGap: "8px",
    display: "grid",
    gridTemplateColumns: "auto auto",
    height: "100%",
    justifyContent: "center",
    padding: "24px",
    rowGap: "24px",
  },
  group: {
    alignItems: "center",
    display: "grid",
    gap: "8px",
    gridColumn: "1 / -1",
    gridTemplateColumns: "subgrid",
  },
  heading: {
    gridColumn: "1 / -1",
    margin: 0,
  },
  label: {
    textAlign: "right",
  },
  input: {
    width: "120px",
  },
});

Promise.all([
  window.ipcRenderer.invoke<ipc.GetSettings>(ipc.getSettings),
  window.ipcRenderer.invoke<ipc.GetSystemInformation>(ipc.getSystemInformation),
]).then(([settings, systemInformation]) => {
  createRoot(document.getElementById("app") as HTMLElement).render(
    <StrictMode>
      <SettingsWindow
        initialSettings={settings}
        systemInformation={systemInformation}
      />
    </StrictMode>,
  );
});
