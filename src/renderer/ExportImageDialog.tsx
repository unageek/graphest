import { Stack } from "@fluentui/react";
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
import { CancelIcon } from "@fluentui/react-icons-mdl2";
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

export interface ExportImageDialogProps {
  abort: () => void;
  dismiss: () => void;
  exportImage: (opts: ExportImageOptions) => Promise<void>;
  opts: ExportImageOptions;
  saveOpts: (opts: ExportImageOptions) => void;
  showSaveDialog: (path: string) => Promise<string | undefined>;
}

const antiAliasingOptions = [
  { key: "1", text: "None" },
  { key: "5", text: "5 × 5" },
  { key: "9", text: "9 × 9" },
  { key: "13", text: "13 × 13" },
  { key: "17", text: "17 × 17" },
  { key: "21", text: "21 × 21" },
  { key: "25", text: "25 × 25" },
];

const useStyles = makeStyles({
  antialiasingDropdown: {
    width: "250px",
  },
  decimalInput: {
    width: "150px",
  },
  integerInput: {
    width: "100px",
  },
});

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
      open={true}
      onOpenChange={(_, { open }) => {
        if (!open && state !== State.Exporting) {
          props.dismiss();
        }
      }}
    >
      <DialogSurface>
        {(() => {
          switch (state) {
            case State.Initial:
              return (
                <form onSubmit={submit}>
                  <DialogBody>
                    <DialogTitle>Export as Image</DialogTitle>

                    <DialogContent
                      style={{
                        alignItems: "baseline",
                        display: "grid",
                        gap: "8px",
                        gridTemplateColumns: "auto auto auto",
                        margin: "8px auto",
                        maxWidth: "fit-content",
                      }}
                    >
                      <Label style={{ gridColumn: "2", padding: 0 }}>
                        Minimum
                      </Label>
                      <Label style={{ gridColumn: "3", padding: 0 }}>
                        Maximum
                      </Label>

                      <Label style={{ textAlign: "right" }}>x:</Label>
                      <Field validationMessage={xMinErrorMessage}>
                        <Input
                          className={styles.decimalInput}
                          onChange={(_, { value }) => {
                            setXMin(value);
                            validateXMin(value);
                          }}
                          value={xMin}
                        />
                      </Field>
                      <Field validationMessage={xMaxErrorMessage}>
                        <Input
                          className={styles.decimalInput}
                          onChange={(_, { value }) => {
                            setXMax(value);
                            validateXMax(value);
                          }}
                          value={xMax}
                        />
                      </Field>

                      <Label style={{ textAlign: "right" }}>y:</Label>
                      <Field validationMessage={yMinErrorMessage}>
                        <Input
                          className={styles.decimalInput}
                          onChange={(_, { value }) => {
                            setYMin(value);
                            validateYMin(value);
                          }}
                          value={yMin}
                        />
                      </Field>
                      <Field validationMessage={yMaxErrorMessage}>
                        <Input
                          className={styles.decimalInput}
                          onChange={(_, { value }) => {
                            setYMax(value);
                            validateYMax(value);
                          }}
                          value={yMax}
                        />
                      </Field>

                      <div style={{ gridColumn: "1 / span 3" }} />

                      <Label style={{ gridColumn: "1", textAlign: "right" }}>
                        Width:
                      </Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "4px",
                          gridColumn: "span 2",
                        }}
                      >
                        <Field validationMessage={widthErrorMessage}>
                          <Input
                            className={styles.integerInput}
                            onChange={(_, { value }) => {
                              setWidth(value);
                              validateWidth(value);
                            }}
                            value={width}
                          />
                        </Field>
                        <Text>pixels</Text>
                      </div>

                      <Label style={{ gridColumn: "1", textAlign: "right" }}>
                        Height:
                      </Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "4px",
                          gridColumn: "span 2",
                        }}
                      >
                        <Field validationMessage={heightErrorMessage}>
                          <Input
                            className={styles.integerInput}
                            onChange={(_, { value }) => {
                              setHeight(value);
                              validateHeight(value);
                            }}
                            value={height}
                          />
                        </Field>
                        <Text>pixels</Text>
                      </div>

                      <div style={{ gridColumn: "1 / span 3" }} />

                      <Label
                        style={{
                          gridColumn: "1",
                          textAlign: "right",
                        }}
                      >
                        Color options:
                      </Label>
                      <Checkbox
                        defaultChecked={opts.transparent}
                        indicator={{
                          style: {
                            display: "inline-block",
                            margin: "2px 8px -5px 0",
                          },
                        }}
                        label={{
                          children: "Transparent background",
                          style: { padding: 0 },
                        }}
                        onChange={(_, { checked }) => {
                          if (checked === undefined || checked === "mixed")
                            return;
                          setOpts({ ...opts, transparent: checked });
                        }}
                        size="large"
                        style={{
                          display: "inline-block",
                          gridColumn: "2 / span 2",
                          padding: "0 0 2px 0",
                        }}
                        title="Make the image background transparent."
                      />

                      <Checkbox
                        defaultChecked={opts.correctAlpha}
                        indicator={{
                          style: {
                            display: "inline-block",
                            margin: "2px 8px -5px 0",
                          },
                        }}
                        label={{
                          children: "Correct alpha composition",
                          style: { padding: 0 },
                        }}
                        onChange={(_, { checked }) => {
                          if (checked === "mixed") return;
                          setOpts({ ...opts, correctAlpha: checked });
                        }}
                        size="large"
                        style={{
                          display: "inline-block",
                          gridColumn: "2 / span 2",
                          padding: "0 0 2px 0",
                        }}
                        title="Perform alpha composition in linear color space."
                      />

                      <div style={{ gridColumn: "1 / span 3" }} />

                      <Label style={{ gridColumn: "1", textAlign: "right" }}>
                        Anti-aliasing:
                      </Label>
                      <Dropdown
                        className={styles.antialiasingDropdown}
                        defaultSelectedOptions={[opts.antiAliasing.toString()]}
                        defaultValue={opts.antiAliasing.toString()}
                        onOptionSelect={(_, { optionValue }) => {
                          if (optionValue === undefined) return;
                          setOpts({
                            ...opts,
                            antiAliasing: Number(optionValue),
                          });
                        }}
                        style={{ gridColumn: "2 / span 2" }}
                      >
                        {antiAliasingOptions.map((option) => (
                          <Option value={option.key}>{option.text}</Option>
                        ))}
                      </Dropdown>

                      <Text style={{ gridColumn: "2 / span 2" }}>
                        Minimum pen size: {approxMinPenSize.toString()} pixel
                        <br />
                        {tilesPerRelation}{" "}
                        {tilesPerRelation > 1 ? "tiles" : "tile"} per relation
                        will be processed.
                      </Text>

                      <div style={{ gridColumn: "1 / span 3" }} />

                      <Label style={{ gridColumn: "1", textAlign: "right" }}>
                        Per-tile timeout:
                      </Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "4px",
                          gridColumn: "span 2",
                        }}
                      >
                        <Field validationMessage={timeoutErrorMessage}>
                          <Input
                            className={styles.integerInput}
                            onChange={(_, { value }) => {
                              setTimeout(value);
                              validateTimeout(value);
                            }}
                            value={timeout}
                          />
                        </Field>
                        <Text>seconds</Text>
                      </div>

                      <div style={{ gridColumn: "1 / span 3" }} />

                      <Label style={{ gridColumn: "1", textAlign: "right" }}>
                        Save as:
                      </Label>
                      <div
                        style={{
                          alignItems: "baseline",
                          display: "flex",
                          flexDirection: "row",
                          gap: "8px",
                          gridColumn: "span 2",
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

            case State.Exporting:
            case State.Exported:
              return (
                <DialogBody>
                  <DialogContent style={{ minWidth: "300px" }}>
                    <Label style={{ padding: "0" }}>
                      {state === State.Exporting ? "Exporting…" : "Exported"}
                    </Label>
                    <Stack
                      horizontal
                      tokens={{ childrenGap: "4px" }}
                      verticalAlign="center"
                    >
                      <Stack.Item grow>
                        <ProgressBar value={progress.progress} />
                      </Stack.Item>
                      {state === State.Exporting && (
                        <Button
                          appearance="subtle"
                          icon={<CancelIcon />}
                          onClick={() => {
                            props.abort();
                            props.dismiss();
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
                  </DialogContent>

                  {state === State.Exported && (
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
