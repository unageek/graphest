import {
  Checkbox,
  DefaultButton,
  Dialog,
  DialogFooter,
  Dropdown,
  IconButton,
  IDropdownOption,
  Label,
  PrimaryButton,
  ProgressIndicator,
  Stack,
  Text,
} from "@fluentui/react";
import { debounce } from "lodash";
import * as React from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { bignum, BigNumber } from "../common/bignumber";
import {
  ExportImageOptions,
  EXPORT_GRAPH_TILE_SIZE,
  MAX_EXPORT_IMAGE_SIZE,
  MAX_EXPORT_TIMEOUT,
} from "../common/exportImage";
import { tryParseBignum, tryParseIntegerInRange } from "../common/parse";
import { useSelector } from "./models/app";
import { SendableTextField } from "./SendableTextField";

export interface ExportImageDialogProps {
  abort: () => void;
  dismiss: () => void;
  exportImage: (opts: ExportImageOptions) => Promise<void>;
  opts: ExportImageOptions;
  saveOpts: (opts: ExportImageOptions) => void;
  showSaveDialog: (path: string) => Promise<string | undefined>;
}

const antiAliasingOptions: IDropdownOption[] = [
  { key: "1", text: "None" },
  { key: "5", text: "5 × 5" },
  { key: "9", text: "9 × 9" },
  { key: "13", text: "13 × 13" },
  { key: "17", text: "17 × 17" },
  { key: "21", text: "21 × 21" },
  { key: "25", text: "25 × 25" },
];

const decimalInputStyles = {
  root: {
    width: "150px",
  },
};

const integerInputStyles = {
  root: {
    width: "100px",
  },
};

const validateRange = (min: BigNumber, max: BigNumber): string | undefined => {
  if (!min.lt(max)) {
    return "Invalid range.";
  }
};

enum State {
  Initial = "initial",
  Exporting = "exporting",
  Exported = "exported",
}

export const ExportImageDialog = (
  props: ExportImageDialogProps
): JSX.Element => {
  const [errors, setErrors] = useState<Set<string>>(new Set());
  const [opts, setOpts] = useState(props.opts);
  const progress = useSelector((s) => s.exportImageProgress);
  const [state, setState] = useState(State.Initial);

  // Field values.
  const [height, setHeight] = useState(opts.height.toString());
  const [timeout, setTimeout] = useState(opts.timeout.toString());
  const [width, setWidth] = useState(opts.width.toString());
  const [xMax, setXMax] = useState(opts.xMax);
  const [xMin, setXMin] = useState(opts.xMin);
  const [yMax, setYMax] = useState(opts.yMax);
  const [yMin, setYMin] = useState(opts.yMin);

  // Field validation error messages.
  const [heightErrorMessage, setHeightErrorMessage] = useState<string>();
  const [timeoutErrorMessage, setTimeoutErrorMessage] = useState<string>();
  const [widthErrorMessage, setWidthErrorMessage] = useState<string>();
  const [xMaxErrorMessage, setXMaxErrorMessage] = useState<string>();
  const [xMinErrorMessage, setXMinErrorMessage] = useState<string>();
  const [yMaxErrorMessage, setYMaxErrorMessage] = useState<string>();
  const [yMinErrorMessage, setYMinErrorMessage] = useState<string>();

  let pathParts = [];
  let separator = "/";
  if (/^([A-Z]:\\|\\\\)/.test(opts.path)) {
    pathParts = opts.path.split("\\");
    separator = "\\";
  } else {
    pathParts = opts.path.split("/");
  }
  const briefPath =
    pathParts.length <= 3
      ? opts.path
      : "…" + separator + pathParts.slice(-2).join(separator);

  const addOrRemoveErrors = useCallback(
    (keys: string[], e?: string): string | undefined => {
      const newErrors = new Set(errors);
      for (const key of keys) {
        if (e !== undefined) {
          newErrors.add(key);
        } else {
          newErrors.delete(key);
        }
      }
      setErrors(newErrors);
      return e;
    },
    [errors]
  );

  const send = useCallback(async () => {
    if (errors.size > 0) return;
    setState(State.Exporting);
    props.saveOpts(opts);
    await props.exportImage(opts);
    setState(State.Exported);
  }, [errors, opts, props]);

  const validateHeight = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseIntegerInRange(value, 1, MAX_EXPORT_IMAGE_SIZE);
        if (result.ok !== undefined) {
          setOpts({ ...opts, height: result.ok });
        }
        setHeightErrorMessage(addOrRemoveErrors(["height"], result.err));
      }, 200),
    [addOrRemoveErrors, opts]
  );

  const validateTimeout = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseIntegerInRange(value, 1, MAX_EXPORT_TIMEOUT);
        if (result.ok !== undefined) {
          setOpts({ ...opts, timeout: result.ok });
        }
        setTimeoutErrorMessage(addOrRemoveErrors(["timeout"], result.err));
      }, 200),
    [addOrRemoveErrors, opts]
  );

  const validateWidth = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseIntegerInRange(value, 1, MAX_EXPORT_IMAGE_SIZE);
        if (result.ok !== undefined) {
          setOpts({ ...opts, width: result.ok });
        }
        setWidthErrorMessage(addOrRemoveErrors(["width"], result.err));
      }, 200),
    [addOrRemoveErrors, opts]
  );

  const validateXMax = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseBignum(value);
        if (result.ok) {
          const rangeError = validateRange(bignum(xMin), result.ok);
          if (!rangeError) {
            setOpts({ ...opts, xMax: value, xMin });
          }
          const error = addOrRemoveErrors(["x-max", "x-min"], rangeError);
          setXMaxErrorMessage(error);
          setXMinErrorMessage(error);
        } else {
          const parseError = result.err;
          setXMaxErrorMessage(addOrRemoveErrors(["x-max"], parseError));
        }
      }, 200),
    [addOrRemoveErrors, opts, xMin]
  );

  const validateXMin = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseBignum(value);
        if (result.ok) {
          const rangeError = validateRange(result.ok, bignum(xMax));
          if (!rangeError) {
            setOpts({ ...opts, xMax, xMin: value });
          }
          const error = addOrRemoveErrors(["x-max", "x-min"], rangeError);
          setXMaxErrorMessage(error);
          setXMinErrorMessage(error);
        } else {
          const parseError = result.err;
          setXMinErrorMessage(addOrRemoveErrors(["x-min"], parseError));
        }
      }, 200),
    [addOrRemoveErrors, opts, xMax]
  );

  const validateYMax = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseBignum(value);
        if (result.ok) {
          const rangeError = validateRange(bignum(yMin), result.ok);
          if (!rangeError) {
            setOpts({ ...opts, yMax: value, yMin });
          }
          const error = addOrRemoveErrors(["y-max", "y-min"], rangeError);
          setYMaxErrorMessage(error);
          setYMinErrorMessage(error);
        } else {
          const parseError = result.err;
          setYMaxErrorMessage(addOrRemoveErrors(["y-max"], parseError));
        }
      }, 200),
    [addOrRemoveErrors, opts, yMin]
  );

  const validateYMin = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseBignum(value);
        if (result.ok) {
          const rangeError = validateRange(result.ok, bignum(yMax));
          if (!rangeError) {
            setOpts({ ...opts, yMax, yMin: value });
          }
          const error = addOrRemoveErrors(["y-max", "y-min"], rangeError);
          setYMaxErrorMessage(error);
          setYMinErrorMessage(error);
        } else {
          const parseError = result.err;
          setYMinErrorMessage(addOrRemoveErrors(["y-min"], parseError));
        }
      }, 200),
    [addOrRemoveErrors, opts, yMax]
  );

  useEffect(() => {
    if (state === State.Exported && progress.messages.length === 0) {
      props.dismiss();
    }
  }, [progress.messages.length, props, state]);

  const tilesPerRelation =
    Math.ceil((opts.antiAliasing * opts.width) / EXPORT_GRAPH_TILE_SIZE) *
    Math.ceil((opts.antiAliasing * opts.height) / EXPORT_GRAPH_TILE_SIZE);

  const minPenSize = bignum(1).div(bignum(opts.antiAliasing));
  const digits = Math.floor(-Math.log10(opts.antiAliasing));
  const scale = bignum(10).pow(digits - 1);
  const approxMinPenSize = minPenSize.div(scale).ceil().times(scale);

  return (
    <Dialog
      dialogContentProps={{
        title: "Export as Image",
        styles:
          state === State.Initial
            ? {}
            : {
                header: { display: "none" },
                inner: { padding: "24px" },
              },
      }}
      hidden={false}
      maxWidth={"100vw"}
      onDismiss={() => {
        if (state !== State.Exporting) {
          props.dismiss();
        }
      }}
      styles={{ main: { minHeight: "0" } }}
    >
      {(() => {
        switch (state) {
          case State.Initial:
            return (
              <>
                <div
                  style={{
                    alignItems: "baseline",
                    display: "grid",
                    gap: "8px",
                    gridTemplateColumns: "auto auto auto",
                  }}
                >
                  <Label style={{ gridColumn: "2", padding: "0" }}>
                    Minimum
                  </Label>
                  <Label style={{ gridColumn: "3", padding: "0" }}>
                    Maximum
                  </Label>

                  <Label style={{ textAlign: "right" }}>x:</Label>
                  <SendableTextField
                    errorMessage={xMinErrorMessage}
                    onChange={(_, value) => {
                      if (value === undefined) return;
                      setXMin(value);
                      validateXMin(value);
                    }}
                    onSend={send}
                    styles={decimalInputStyles}
                    value={xMin}
                  />
                  <SendableTextField
                    errorMessage={xMaxErrorMessage}
                    onChange={(_, value) => {
                      if (value === undefined) return;
                      setXMax(value);
                      validateXMax(value);
                    }}
                    onSend={send}
                    styles={decimalInputStyles}
                    value={xMax}
                  />

                  <Label style={{ textAlign: "right" }}>y:</Label>
                  <SendableTextField
                    errorMessage={yMinErrorMessage}
                    onChange={(_, value) => {
                      if (value === undefined) return;
                      setYMin(value);
                      validateYMin(value);
                    }}
                    onSend={send}
                    styles={decimalInputStyles}
                    value={yMin}
                  />
                  <SendableTextField
                    errorMessage={yMaxErrorMessage}
                    onChange={(_, value) => {
                      if (value === undefined) return;
                      setYMax(value);
                      validateYMax(value);
                    }}
                    onSend={send}
                    styles={decimalInputStyles}
                    value={yMax}
                  />

                  <div />

                  <Label style={{ gridColumn: "1", textAlign: "right" }}>
                    Width:
                  </Label>
                  <Stack
                    horizontal
                    style={{ gridColumn: "span 2" }}
                    tokens={{ childrenGap: "4px" }}
                    verticalAlign="baseline"
                  >
                    <SendableTextField
                      errorMessage={widthErrorMessage}
                      onChange={(_, value) => {
                        if (value === undefined) return;
                        setWidth(value);
                        validateWidth(value);
                      }}
                      onSend={send}
                      styles={integerInputStyles}
                      value={width}
                    />
                    <Text>pixels</Text>
                  </Stack>

                  <Label style={{ gridColumn: "1", textAlign: "right" }}>
                    Height:
                  </Label>
                  <Stack
                    horizontal
                    style={{ gridColumn: "span 2" }}
                    tokens={{ childrenGap: "4px" }}
                    verticalAlign="baseline"
                  >
                    <SendableTextField
                      errorMessage={heightErrorMessage}
                      onChange={(_, value) => {
                        if (value === undefined) return;
                        setHeight(value);
                        validateHeight(value);
                      }}
                      onSend={send}
                      styles={integerInputStyles}
                      value={height}
                    />
                    <Text>pixels</Text>
                  </Stack>

                  <Checkbox
                    defaultChecked={opts.transparent}
                    label="Transparent background"
                    onChange={(_, checked) => {
                      if (checked === undefined) return;
                      setOpts({ ...opts, transparent: checked });
                    }}
                    styles={{ root: { gridColumn: "2 / span 2" } }}
                    title="Make the image background transparent."
                  />

                  <Checkbox
                    defaultChecked={opts.correctAlpha}
                    label="Correct alpha composition"
                    onChange={(_, checked) => {
                      if (checked === undefined) return;
                      setOpts({ ...opts, correctAlpha: checked });
                    }}
                    styles={{ root: { gridColumn: "2 / span 2" } }}
                    title="Perform alpha composition in linear color space."
                  />

                  <div style={{ gridColumn: "1" }} />

                  <Label style={{ gridColumn: "1", textAlign: "right" }}>
                    Anti-aliasing:
                  </Label>
                  <Dropdown
                    defaultSelectedKey={opts.antiAliasing.toString()}
                    onChange={(_, option) => {
                      if (option === undefined) return;
                      setOpts({ ...opts, antiAliasing: Number(option.key) });
                    }}
                    options={antiAliasingOptions}
                    styles={integerInputStyles}
                  />

                  <Stack style={{ gridColumn: "2 / span 2" }}>
                    <Text>
                      Minimum pen size: {approxMinPenSize.toString()} pixel
                    </Text>
                    <Text>
                      {tilesPerRelation}{" "}
                      {tilesPerRelation > 1 ? "tiles" : "tile"} per relation
                      will be processed.
                    </Text>
                  </Stack>

                  <div style={{ gridColumn: "1" }} />

                  <Label style={{ gridColumn: "1", textAlign: "right" }}>
                    Per-tile timeout:
                  </Label>
                  <Stack
                    horizontal
                    style={{ gridColumn: "span 2" }}
                    tokens={{ childrenGap: "4px" }}
                    verticalAlign="baseline"
                  >
                    <SendableTextField
                      errorMessage={timeoutErrorMessage}
                      onChange={(_, value) => {
                        if (value === undefined) return;
                        setTimeout(value);
                        validateTimeout(value);
                      }}
                      onSend={send}
                      styles={integerInputStyles}
                      value={timeout}
                    />
                    <Text>seconds</Text>
                  </Stack>

                  <div style={{ gridColumn: "1" }} />

                  <Label style={{ gridColumn: "1", textAlign: "right" }}>
                    Save as:
                  </Label>
                  <Stack
                    horizontal
                    style={{ gridColumn: "span 2" }}
                    tokens={{ childrenGap: "8px" }}
                    verticalAlign="baseline"
                  >
                    <Text
                      styles={{
                        root: {
                          maxWidth: "200px",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        },
                      }}
                    >
                      {briefPath}
                    </Text>
                    <DefaultButton
                      onClick={async () => {
                        const path = await props.showSaveDialog(opts.path);
                        // `path` can be an empty string if the user cancels the dialog on macOS.
                        if (path) {
                          setOpts({ ...opts, path });
                        }
                      }}
                      text="Change…"
                    />
                  </Stack>
                </div>

                <DialogFooter>
                  <DefaultButton onClick={props.dismiss} text="Cancel" />
                  <PrimaryButton
                    disabled={errors.size > 0}
                    onClick={send}
                    text="Export"
                  />
                </DialogFooter>
              </>
            );

          case State.Exporting:
          case State.Exported:
            return (
              <>
                <div style={{ minWidth: "300px" }}>
                  <Label style={{ padding: "0" }}>
                    {state === State.Exporting ? "Exporting…" : "Exported"}
                  </Label>
                  <Stack
                    horizontal
                    tokens={{ childrenGap: "4px" }}
                    verticalAlign="center"
                  >
                    <Stack.Item grow>
                      <ProgressIndicator percentComplete={progress.progress} />
                    </Stack.Item>
                    {state === State.Exporting && (
                      <IconButton
                        iconProps={{ iconName: "Cancel" }}
                        onClick={() => {
                          props.abort();
                          props.dismiss();
                        }}
                        styles={{
                          root: {
                            height: "24px",
                            margin: "-4px",
                            width: "24px",
                          },
                        }}
                        title="Cancel"
                      />
                    )}
                  </Stack>
                  {progress.messages.map((message, index) => (
                    <Text block key={index}>
                      {message}
                    </Text>
                  ))}
                </div>

                {state === State.Exported && (
                  <DialogFooter>
                    <DefaultButton onClick={props.dismiss} text="Done" />
                  </DialogFooter>
                )}
              </>
            );
        }
      })()}
    </Dialog>
  );
};
