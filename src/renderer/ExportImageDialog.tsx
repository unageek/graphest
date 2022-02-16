import {
  ComboBox,
  DefaultButton,
  Dialog,
  DialogFooter,
  IComboBoxOption,
  Label,
  PrimaryButton,
  Spinner,
  SpinnerSize,
  Stack,
  Text,
  TextField,
} from "@fluentui/react";
import * as React from "react";
import { useState } from "react";
import { bignum, BigNumber } from "../common/bignumber";
import { MAX_EXPORT_IMAGE_SIZE, MAX_EXPORT_TIMEOUT } from "../common/constants";
import { ExportImageOptions } from "../common/exportImageOptions";
import { err, ok, Result } from "../common/result";

export interface ExportImageDialogProps {
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
  // { key: "11", text: "11 × 11" },
  // { key: "13", text: "13 × 13" },
  // { key: "15", text: "15 × 15" },
  // { key: "17", text: "17 × 17" },
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

const parseBignum = (value?: string): Result<BigNumber, string | undefined> => {
  const val = bignum(value ?? "");
  if (val.isFinite()) {
    return ok(val);
  } else {
    return err("A number is required.");
  }
};

const parseImageSize = (value?: string): Result<number, string | undefined> => {
  const val = Number(value);
  if (Number.isSafeInteger(val) && val > 0 && val <= MAX_EXPORT_IMAGE_SIZE) {
    return ok(val);
  } else {
    return err(`Image size must be between 1 and ${MAX_EXPORT_IMAGE_SIZE}.`);
  }
};

const parseTimeout = (value?: string): Result<number, string | undefined> => {
  const val = Number(value);
  if (Number.isSafeInteger(val) && val > 0 && val <= MAX_EXPORT_TIMEOUT) {
    return ok(val);
  } else {
    return err(`Timeout must be between 1 and ${MAX_EXPORT_TIMEOUT}.`);
  }
};

const textStyles = {
  root: {
    padding: "5px 0px",
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
  const [errors, setErrors] = useState<string[]>([]);
  const [exporting, setExporting] = useState(false);
  const [opts, setOpts] = useState(props.opts);

  const pathParts = opts.path.split(new RegExp("[\\/]"));
  const briefPath =
    pathParts.length <= 2 ? pathParts : "…/" + pathParts.slice(-2).join("/");

  function addOrRemoveError<T>(key: string, e?: T): T | undefined {
    if (e !== undefined) {
      setErrors([...errors, key]);
      return e;
    } else {
      setErrors(errors.filter((e) => e !== key));
      return undefined;
    }
  }

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
        <Spinner label="Exporting…" size={SpinnerSize.large} />
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gap: "10px",
              gridTemplateColumns: "auto auto auto",
            }}
          >
            <Label style={{ gridColumn: "2" }}>Min</Label>
            <Label style={{ gridColumn: "3" }}>Max</Label>

            <Label style={{ textAlign: "right" }}>x</Label>
            <TextField
              defaultValue={opts.xMin}
              onChange={(_, value) => {
                const val = parseBignum(value).ok;
                if (val !== undefined) {
                  setOpts({ ...opts, xMin: val.toString() });
                }
              }}
              onGetErrorMessage={(value) => {
                const e =
                  parseBignum(value).err ??
                  validateRange(bignum(value), bignum(opts.xMax));
                return addOrRemoveError("x-min", e);
              }}
              styles={decimalInputStyles}
            />
            <TextField
              defaultValue={opts.xMax}
              onChange={(_, value) => {
                const val = parseBignum(value).ok;
                if (val !== undefined) {
                  setOpts({ ...opts, xMax: val.toString() });
                }
              }}
              onGetErrorMessage={(value) => {
                const e =
                  parseBignum(value).err ??
                  validateRange(bignum(opts.xMin), bignum(value));
                return addOrRemoveError("x-max", e);
              }}
              styles={decimalInputStyles}
            />

            <Label style={{ textAlign: "right" }}>y</Label>
            <TextField
              defaultValue={opts.yMin}
              onChange={(_, value) => {
                if (!value) return;
                const val = bignum(value);
                if (val.isFinite()) {
                  setOpts({ ...opts, yMin: val.toString() });
                }
              }}
              onGetErrorMessage={(value) => {
                const e =
                  parseBignum(value).err ??
                  validateRange(bignum(value), bignum(opts.yMax));
                return addOrRemoveError("y-min", e);
              }}
              styles={decimalInputStyles}
            />
            <TextField
              defaultValue={opts.yMax}
              onChange={(_, value) => {
                if (!value) return;
                const val = bignum(value);
                if (val.isFinite()) {
                  setOpts({ ...opts, yMax: val.toString() });
                }
              }}
              onGetErrorMessage={(value) => {
                const e =
                  parseBignum(value).err ??
                  validateRange(bignum(opts.yMin), bignum(value));
                return addOrRemoveError("y-max", e);
              }}
              styles={decimalInputStyles}
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
                  const val = parseImageSize(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, width: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = parseImageSize(value).err;
                  return addOrRemoveError("width", e);
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
                  const val = parseImageSize(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, height: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = parseImageSize(value).err;
                  return addOrRemoveError("height", e);
                }}
                styles={integerInputStyles}
              />
              <Text styles={textStyles}>pixels</Text>
            </Stack>

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Anti‑aliasing
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

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Timeout
            </Label>
            <Stack
              horizontal
              style={{ gridColumn: "span 2" }}
              tokens={{ childrenGap: "5" }}
            >
              <TextField
                defaultValue={opts.timeout.toString()}
                onChange={(_, value) => {
                  const val = parseTimeout(value).ok;
                  if (val !== undefined) {
                    setOpts({ ...opts, timeout: val });
                  }
                }}
                onGetErrorMessage={(value) => {
                  const e = parseTimeout(value).err;
                  return addOrRemoveError("timeout", e);
                }}
                styles={integerInputStyles}
              />
              <Text styles={textStyles}>seconds</Text>
            </Stack>

            <div style={{ gridColumn: "1" }} />

            <Label style={{ gridColumn: "1", textAlign: "right" }}>
              Save as
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
              disabled={errors.length > 0}
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
