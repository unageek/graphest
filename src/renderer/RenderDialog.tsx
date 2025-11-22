import {
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  Dropdown,
  Field,
  Input,
  Label,
  makeStyles,
  Option,
  ProgressBar,
  Text,
} from "@fluentui/react-components";
import { DismissRegular } from "@fluentui/react-icons";
import { debounce } from "lodash";
import * as React from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { bignum, BigNumber } from "../common/bignumber";
import {
  EXPORT_GRAPH_TILE_SIZE,
  ExportImageOptions,
  MAX_EXPORT_IMAGE_SIZE,
  MAX_EXPORT_TIMEOUT,
} from "../common/exportImage";
import { tryParseBignum, tryParseIntegerInRange } from "../common/parse";
import { useSelector } from "./models/app";

export interface RenderDialogProps {
  abort: () => void;
  dismiss: () => void;
  exportImage: (opts: ExportImageOptions) => Promise<void>;
  opts: ExportImageOptions;
  saveOpts: (opts: ExportImageOptions) => void;
  showSaveDialog: (path: string) => Promise<string | undefined>;
}

interface AntiAliasingOption {
  value: string;
  text: string;
}

const antiAliasingOptions: AntiAliasingOption[] = [
  { value: "1", text: "None" },
  { value: "5", text: "5 × 5" },
  { value: "9", text: "9 × 9" },
  { value: "13", text: "13 × 13" },
  { value: "17", text: "17 × 17" },
  { value: "21", text: "21 × 21" },
  { value: "25", text: "25 × 25" },
];

const antiAliasingOptionText: Map<string, string> = new Map(
  antiAliasingOptions.map((o) => [o.value, o.text])
);

const useStyles = makeStyles({
  antialiasingDropdown: {
    minWidth: "unset",
    width: "150px",
  },
  checkbox: {
    display: "inline-block",
  },
  checkboxIndicator: {
    display: "inline-flex",
    margin: "0 8px -3px 0",
    "& > svg": {
      position: "absolute",
    },
  },
  checkboxLabel: {
    padding: 0,
  },
  input: {
    width: "150px",
  },
});

const validateRange = (min: BigNumber, max: BigNumber): string | undefined => {
  if (!min.lt(max)) {
    return "Invalid range.";
  }
};

enum State {
  Initial = "initial",
  Processing = "processing",
  Complete = "complete",
}

export const RenderDialog = (props: RenderDialogProps): JSX.Element => {
  const [errors, setErrors] = useState<Set<string>>(new Set());
  const [opts, setOpts] = useState(props.opts);
  const progress = useSelector((s) => s.exportImageProgress);
  const [state, setState] = useState(State.Initial);
  const styles = useStyles();

  // Input values.
  const [height, setHeight] = useState(opts.height.toString());
  const [timeout, setTimeout] = useState(opts.timeout.toString());
  const [width, setWidth] = useState(opts.width.toString());
  const [xMax, setXMax] = useState(opts.xMax);
  const [xMin, setXMin] = useState(opts.xMin);
  const [yMax, setYMax] = useState(opts.yMax);
  const [yMin, setYMin] = useState(opts.yMin);

  // Input validation error messages.
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

  const submit = useCallback(async () => {
    if (errors.size > 0) return;
    setState(State.Processing);
    props.saveOpts(opts);
    await props.exportImage(opts);
    setState(State.Complete);
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
        if (result.ok !== undefined) {
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
        if (result.ok !== undefined) {
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
        if (result.ok !== undefined) {
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
        if (result.ok !== undefined) {
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
    if (state === State.Complete && progress.messages.length === 0) {
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
      open={true}
      onOpenChange={(_, { open }) => {
        if (!open && state !== State.Processing) {
          props.dismiss();
        }
      }}
    >
      <DialogSurface style={{ width: "fit-content" }}>
        {(() => {
          switch (state) {
            case State.Initial:
              return (
                <form onSubmit={submit}>
                  <DialogBody>
                    <DialogTitle>Render</DialogTitle>

                    <DialogContent
                      style={{
                        alignItems: "baseline",
                        display: "grid",
                        gap: "8px",
                        gridTemplateColumns: "auto auto",
                        margin: "8px auto",
                        maxWidth: "fit-content",
                      }}
                    >
                      <Label style={{ textAlign: "right" }}>x:</Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "8px",
                        }}
                      >
                        <Field validationMessage={xMinErrorMessage}>
                          <Input
                            className={styles.input}
                            onChange={(_, { value }) => {
                              setXMin(value);
                              validateXMin(value);
                            }}
                            value={xMin}
                          />
                        </Field>
                        <Text>…</Text>
                        <Field validationMessage={xMaxErrorMessage}>
                          <Input
                            className={styles.input}
                            onChange={(_, { value }) => {
                              setXMax(value);
                              validateXMax(value);
                            }}
                            value={xMax}
                          />
                        </Field>
                      </div>

                      <Label style={{ textAlign: "right" }}>y:</Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "8px",
                        }}
                      >
                        <Field validationMessage={yMinErrorMessage}>
                          <Input
                            className={styles.input}
                            onChange={(_, { value }) => {
                              setYMin(value);
                              validateYMin(value);
                            }}
                            value={yMin}
                          />
                        </Field>
                        <Text>…</Text>
                        <Field validationMessage={yMaxErrorMessage}>
                          <Input
                            className={styles.input}
                            onChange={(_, { value }) => {
                              setYMax(value);
                              validateYMax(value);
                            }}
                            value={yMax}
                          />
                        </Field>
                      </div>

                      <div style={{ gridColumn: "1 / span 2" }} />

                      <Label style={{ textAlign: "right" }}>Width:</Label>
                      <Field validationMessage={widthErrorMessage}>
                        <Input
                          className={styles.input}
                          contentAfter={<Text>pixels</Text>}
                          onChange={(_, { value }) => {
                            setWidth(value);
                            validateWidth(value);
                          }}
                          value={width}
                        />
                      </Field>

                      <Label style={{ textAlign: "right" }}>Height:</Label>
                      <Field validationMessage={heightErrorMessage}>
                        <Input
                          className={styles.input}
                          contentAfter={<Text>pixels</Text>}
                          onChange={(_, { value }) => {
                            setHeight(value);
                            validateHeight(value);
                          }}
                          value={height}
                        />
                      </Field>

                      <div style={{ gridColumn: "1 / span 2" }} />

                      <Label
                        style={{
                          textAlign: "right",
                        }}
                      >
                        Color options:
                      </Label>
                      <Checkbox
                        className={styles.checkbox}
                        defaultChecked={opts.transparent}
                        indicator={{
                          className: styles.checkboxIndicator,
                        }}
                        label={{
                          children: "Transparent background",
                          className: styles.checkboxLabel,
                        }}
                        onChange={(_, { checked }) => {
                          if (checked === undefined || checked === "mixed")
                            return;
                          setOpts({ ...opts, transparent: checked });
                        }}
                        style={{ gridColumn: "2" }}
                        title="Make the image background transparent."
                      />

                      <Checkbox
                        className={styles.checkbox}
                        defaultChecked={opts.correctAlpha}
                        indicator={{
                          className: styles.checkboxIndicator,
                        }}
                        label={{
                          children: "Correct alpha composition",
                          className: styles.checkboxLabel,
                        }}
                        onChange={(_, { checked }) => {
                          if (checked === "mixed") return;
                          setOpts({ ...opts, correctAlpha: checked });
                        }}
                        style={{ gridColumn: "2" }}
                        title="Perform alpha composition in linear color space."
                      />

                      <div style={{ gridColumn: "1 / span 2" }} />

                      <Label style={{ textAlign: "right" }}>
                        Anti-aliasing:
                      </Label>
                      <Dropdown
                        className={styles.antialiasingDropdown}
                        defaultSelectedOptions={[opts.antiAliasing.toString()]}
                        defaultValue={antiAliasingOptionText.get(
                          opts.antiAliasing.toString()
                        )}
                        listbox={{
                          className: styles.antialiasingDropdown,
                        }}
                        onOptionSelect={(_, { optionValue }) => {
                          if (optionValue === undefined) return;
                          setOpts({
                            ...opts,
                            antiAliasing: Number(optionValue),
                          });
                        }}
                      >
                        {antiAliasingOptions.map((o) => (
                          <Option value={o.value}>{o.text}</Option>
                        ))}
                      </Dropdown>

                      <Text style={{ gridColumn: "2" }}>
                        Minimum pen size: {approxMinPenSize.toString()} pixel
                        <br />
                        {tilesPerRelation}{" "}
                        {tilesPerRelation > 1 ? "tiles" : "tile"} per relation
                        will be processed.
                      </Text>

                      <div style={{ gridColumn: "1 / span 2" }} />

                      <Label style={{ textAlign: "right" }}>
                        Per-tile timeout:
                      </Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "8px",
                        }}
                      >
                        <Field validationMessage={timeoutErrorMessage}>
                          <Input
                            className={styles.input}
                            contentAfter={<Text>seconds</Text>}
                            onChange={(_, { value }) => {
                              setTimeout(value);
                              validateTimeout(value);
                            }}
                            value={timeout}
                          />
                        </Field>
                      </div>

                      <div style={{ gridColumn: "1 / span 2" }} />

                      <Label style={{ textAlign: "right" }}>Save as:</Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "8px",
                        }}
                      >
                        <Text
                          style={{
                            maxWidth: "200px",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {briefPath}
                        </Text>
                        <Button
                          onClick={async () => {
                            const path = await props.showSaveDialog(opts.path);
                            // `path` can be an empty string if the user cancels the dialog on macOS.
                            if (path) {
                              setOpts({ ...opts, path });
                            }
                          }}
                        >
                          Change…
                        </Button>
                      </div>
                    </DialogContent>

                    <DialogActions>
                      <Button onClick={props.dismiss}>Cancel</Button>
                      <Button
                        appearance="primary"
                        disabled={errors.size > 0}
                        type="submit"
                      >
                        Export
                      </Button>
                    </DialogActions>
                  </DialogBody>
                </form>
              );

            case State.Processing:
            case State.Complete:
              return (
                <DialogBody>
                  <DialogContent style={{ minWidth: "300px" }}>
                    <Text>
                      {state === State.Processing ? "Processing…" : "Complete"}
                    </Text>
                    <div
                      style={{
                        alignItems: "center",
                        display: "flex",
                        flexDirection: "row",
                        gap: "8px",
                        height: "30px",
                      }}
                    >
                      <ProgressBar
                        style={{ flexGrow: 1 }}
                        value={progress.progress}
                      />
                      {state === State.Processing && (
                        <Button
                          appearance="subtle"
                          icon={<DismissRegular />}
                          onClick={() => {
                            props.abort();
                            props.dismiss();
                          }}
                          title="Cancel"
                        />
                      )}
                    </div>
                    {progress.messages.map((message, index) => (
                      <Text block key={index}>
                        {message}
                      </Text>
                    ))}
                  </DialogContent>

                  {state === State.Complete && (
                    <DialogActions>
                      <Button onClick={props.dismiss}>Done</Button>
                    </DialogActions>
                  )}
                </DialogBody>
              );
          }
        })()}
      </DialogSurface>
    </Dialog>
  );
};
