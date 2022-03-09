import {
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
import { useCallback, useMemo, useState } from "react";
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
  { key: "3", text: "3 × 3" },
  { key: "5", text: "5 × 5" },
  { key: "7", text: "7 × 7" },
  { key: "9", text: "9 × 9" },
  { key: "11", text: "11 × 11" },
  { key: "13", text: "13 × 13" },
  { key: "15", text: "15 × 15" },
  { key: "17", text: "17 × 17" },
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

export const ExportImageDialog = (
  props: ExportImageDialogProps
): JSX.Element => {
  const [errors, setErrors] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState(false);
  const [opts, setOpts] = useState(props.opts);
  const progress = useSelector((s) => s.exportImageProgress);

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
    setExporting(true);
    props.saveOpts(opts);
    await props.exportImage(opts);
    props.dismiss();
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

  const tilesPerRelation =
    Math.ceil((opts.antiAliasing * opts.width) / EXPORT_GRAPH_TILE_SIZE) *
    Math.ceil((opts.antiAliasing * opts.height) / EXPORT_GRAPH_TILE_SIZE);

  return (
    <Dialog
      dialogContentProps={{
        title: exporting ? "" : "Export as Image",
      }}
      hidden={false}
      maxWidth={"100vw"}
      modalProps={{
        isModeless: true,
      }}
      onDismiss={() => {
        if (!exporting) {
          props.dismiss();
        }
      }}
    >
      {exporting ? (
        <div
          style={{
            display: "grid",
            gap: "16px",
            gridTemplateRows: "auto auto",
          }}
        >
          <div
            style={{
              alignItems: "center",
              display: "grid",
              gap: "4px",
              gridTemplateColumns: "1fr auto",
            }}
          >
            <ProgressIndicator
              label="Exporting…"
              percentComplete={progress.progress}
            />
            <IconButton
              iconProps={{ iconName: "Cancel" }}
              onClick={() => {
                props.abort();
                props.dismiss();
              }}
              title="Cancel"
            />
          </div>
          <img
            style={{ gridRow: "2" }}
            width="256"
            height="256"
            src={progress.lastUrl}
          />
          <div
            style={{ gridRow: "3", minHeight: "1em", whiteSpace: "pre-wrap" }}
          >
            {progress.lastStderr}
          </div>
        </div>
      ) : (
        <>
          <div
            style={{
              alignItems: "baseline",
              display: "grid",
              gap: "8px",
              gridTemplateColumns: "auto auto auto",
            }}
          >
            <Label style={{ gridColumn: "2", padding: "0" }}>Minimum</Label>
            <Label style={{ gridColumn: "3", padding: "0" }}>Maximum</Label>

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
              style={{ alignItems: "baseline", gridColumn: "span 2" }}
              tokens={{ childrenGap: "4px" }}
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
              style={{ alignItems: "baseline", gridColumn: "span 2" }}
              tokens={{ childrenGap: "4px" }}
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

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Anti-aliasing:
            </Label>
            <Dropdown
              defaultSelectedKey={opts.antiAliasing.toString()}
              onChange={(_, option) => {
                if (option) {
                  setOpts({ ...opts, antiAliasing: Number(option.key) });
                }
              }}
              options={antiAliasingOptions}
              styles={integerInputStyles}
            />

            <div style={{ gridColumn: "1" }} />

            <Text style={{ gridColumn: "span 2" }}>
              {tilesPerRelation} {tilesPerRelation > 1 ? "tiles" : "tile"} per
              relation will be processed.
            </Text>
            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Per-tile timeout:
            </Label>
            <Stack
              horizontal
              style={{ alignItems: "baseline", gridColumn: "span 2" }}
              tokens={{ childrenGap: "4px" }}
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
              <Text>milliseconds</Text>
            </Stack>

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Save as:
            </Label>
            <Stack
              horizontal
              style={{ alignItems: "baseline", gridColumn: "span 2" }}
              tokens={{ childrenGap: "8px" }}
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
            <DefaultButton onClick={props.dismiss} text="Close" />
            <PrimaryButton
              disabled={errors.size > 0}
              onClick={send}
              text="Export"
            />
          </DialogFooter>
        </>
      )}
    </Dialog>
  );
};
