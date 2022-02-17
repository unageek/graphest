import {
  ComboBox,
  DefaultButton,
  Dialog,
  DialogFooter,
  IComboBoxOption,
  IconButton,
  Label,
  PrimaryButton,
  ProgressIndicator,
  Stack,
  Text,
  TextField,
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
import { err, ok, Result } from "../common/result";
import { useSelector } from "./models/app";

export interface ExportImageDialogProps {
  abort: () => void;
  dismiss: () => void;
  exportImage: (opts: ExportImageOptions) => Promise<void>;
  openSaveDialog: (path: string) => Promise<string | undefined>;
  opts: ExportImageOptions;
  saveOpts: (opts: ExportImageOptions) => void;
}

const antiAliasingOptions: IComboBoxOption[] = [
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

const textStyles = {
  root: {
    padding: "5px 0px",
  },
};

const tryParseBignum = (
  value?: string
): Result<BigNumber, string | undefined> => {
  const val = bignum(value ?? "");
  if (val.isFinite()) {
    return ok(val);
  } else {
    return err("A number is required.");
  }
};

const tryParseImageSize = (
  value?: string
): Result<number, string | undefined> => {
  const val = Number(value);
  if (Number.isSafeInteger(val) && val > 0 && val <= MAX_EXPORT_IMAGE_SIZE) {
    return ok(val);
  } else {
    return err(`Image size must be between 1 and ${MAX_EXPORT_IMAGE_SIZE}.`);
  }
};

const tryParseTimeout = (
  value?: string
): Result<number, string | undefined> => {
  const val = Number(value);
  if (Number.isSafeInteger(val) && val > 0 && val <= MAX_EXPORT_TIMEOUT) {
    return ok(val);
  } else {
    return err(`Timeout must be between 1 and ${MAX_EXPORT_TIMEOUT}.`);
  }
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

  // These properties are correlated, thus we need to maintain them.
  const [xMax, setXMax] = useState(opts.xMax);
  const [xMaxErrorMessage, setXMaxErrorMessage] = useState<string>();
  const [xMin, setXMin] = useState(opts.xMin);
  const [xMinErrorMessage, setXMinErrorMessage] = useState<string>();
  const [yMax, setYMax] = useState(opts.yMax);
  const [yMaxErrorMessage, setYMaxErrorMessage] = useState<string>();
  const [yMin, setYMin] = useState(opts.yMin);
  const [yMinErrorMessage, setYMinErrorMessage] = useState<string>();

  const pathParts = opts.path.split(new RegExp("[\\/]"));
  const briefPath =
    pathParts.length <= 2 ? pathParts : "…/" + pathParts.slice(-2).join("/");

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

  const validateXMax = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseBignum(value);
        if (result.ok) {
          const rangeError = validateRange(bignum(xMin), result.ok);
          if (!rangeError) {
            setOpts({ ...opts, xMax: value, xMin });
          }
          const errors = addOrRemoveErrors(["x-max", "x-min"], rangeError);
          setXMaxErrorMessage(errors);
          setXMinErrorMessage(errors);
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
          const errors = addOrRemoveErrors(["x-max", "x-min"], rangeError);
          setXMaxErrorMessage(errors);
          setXMinErrorMessage(errors);
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
          const errors = addOrRemoveErrors(["y-max", "y-min"], rangeError);
          setYMaxErrorMessage(errors);
          setYMinErrorMessage(errors);
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
          const errors = addOrRemoveErrors(["y-max", "y-min"], rangeError);
          setYMaxErrorMessage(errors);
          setYMinErrorMessage(errors);
        } else {
          const parseError = result.err;
          setYMinErrorMessage(addOrRemoveErrors(["y-min"], parseError));
        }
      }, 200),
    [addOrRemoveErrors, opts, yMax]
  );

  const tilesPerGraph =
    opts.antiAliasing ** 2 *
    Math.ceil(opts.width / EXPORT_GRAPH_TILE_SIZE) *
    Math.ceil(opts.height / EXPORT_GRAPH_TILE_SIZE);

  return (
    <Dialog
      dialogContentProps={{
        title: exporting ? "" : "Export to Image",
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
            gap: "10px",
            gridTemplateRows: "auto auto",
          }}
        >
          <div
            style={{
              alignItems: "center",
              display: "grid",
              gap: "10px",
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
              display: "grid",
              gap: "10px",
              gridTemplateColumns: "auto auto auto",
            }}
          >
            <Label style={{ gridColumn: "2", padding: "0" }}>Minimum</Label>
            <Label style={{ gridColumn: "3", padding: "0" }}>Maximum</Label>

            <Label style={{ textAlign: "right" }}>x</Label>
            <TextField
              errorMessage={xMinErrorMessage}
              onChange={(_, value) => {
                if (value === undefined) return;
                setXMin(value);
                validateXMin(value);
              }}
              styles={decimalInputStyles}
              value={xMin}
            />
            <TextField
              errorMessage={xMaxErrorMessage}
              onChange={(_, value) => {
                if (value === undefined) return;
                setXMax(value);
                validateXMax(value);
              }}
              styles={decimalInputStyles}
              value={xMax}
            />

            <Label style={{ textAlign: "right" }}>y</Label>
            <TextField
              errorMessage={yMinErrorMessage}
              onChange={(_, value) => {
                if (value === undefined) return;
                setYMin(value);
                validateYMin(value);
              }}
              styles={decimalInputStyles}
              value={yMin}
            />
            <TextField
              errorMessage={yMaxErrorMessage}
              onChange={(_, value) => {
                if (value === undefined) return;
                setYMax(value);
                validateYMax(value);
              }}
              styles={decimalInputStyles}
              value={yMax}
            />

            <div />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>Width</Label>
            <Stack
              horizontal
              style={{ gridColumn: "span 2" }}
              tokens={{ childrenGap: "5" }}
            >
              <TextField
                defaultValue={opts.width.toString()}
                onChange={(_, value) => {
                  const val = tryParseImageSize(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, width: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = tryParseImageSize(value).err;
                  return addOrRemoveErrors(["width"], e);
                }}
                styles={integerInputStyles}
              />
              <Text styles={textStyles}>pixels</Text>
            </Stack>

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Height
            </Label>
            <Stack
              horizontal
              style={{ gridColumn: "span 2" }}
              tokens={{ childrenGap: "5" }}
            >
              <TextField
                defaultValue={opts.height.toString()}
                onChange={(_, value) => {
                  const val = tryParseImageSize(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, height: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = tryParseImageSize(value).err;
                  return addOrRemoveErrors(["height"], e);
                }}
                styles={integerInputStyles}
              />
              <Text styles={textStyles}>pixels</Text>
            </Stack>

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Anti-Aliasing
            </Label>
            <ComboBox
              defaultSelectedKey={opts.antiAliasing.toString()}
              onChange={(_, option?: IComboBoxOption) => {
                if (option) {
                  setOpts({ ...opts, antiAliasing: Number(option.key) });
                }
              }}
              options={antiAliasingOptions}
              styles={{ callout: { width: "100px" }, root: { width: "100px" } }}
            />

            <div style={{ gridColumn: "1" }} />

            <Text style={{ gridColumn: "span 2" }}>
              {tilesPerGraph} {tilesPerGraph > 1 ? "tiles" : "tile"} per
              relation will be processed.
            </Text>
            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Per-tile Timeout
            </Label>
            <Stack
              horizontal
              style={{ gridColumn: "span 2" }}
              tokens={{ childrenGap: "5" }}
            >
              <TextField
                defaultValue={opts.timeout.toString()}
                onChange={(_, value) => {
                  const val = tryParseTimeout(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, timeout: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = tryParseTimeout(value).err;
                  return addOrRemoveErrors(["timeout"], e);
                }}
                styles={integerInputStyles}
              />
              <Text styles={textStyles}>milliseconds</Text>
            </Stack>

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Save As
            </Label>
            <Stack
              horizontal
              style={{ gridColumn: "span 2" }}
              tokens={{ childrenGap: "10px" }}
            >
              <Text styles={textStyles}>{briefPath}</Text>
              <DefaultButton
                onClick={async () => {
                  const path = await props.openSaveDialog(opts.path);
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
              onClick={async () => {
                setExporting(true);
                props.saveOpts(opts);
                await props.exportImage(opts);
                props.dismiss();
              }}
              text="Export"
            />
          </DialogFooter>
        </>
      )}
    </Dialog>
  );
};
