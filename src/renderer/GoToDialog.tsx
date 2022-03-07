import {
  DefaultButton,
  Dialog,
  DialogFooter,
  Label,
  PrimaryButton,
  SpinButton,
  TextField,
} from "@fluentui/react";
import * as React from "react";
import { useCallback, useState } from "react";
import { BASE_ZOOM_LEVEL } from "../common/constants";
import { tryParseNumber } from "../common/parse";

export interface GoToDialogProps {
  dismiss: () => void;
  goTo: (center: [number, number], zoomLevel: number) => void;
  center: [number, number];
  zoomLevel: number;
}

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

export const GoToDialog = (props: GoToDialogProps): JSX.Element => {
  const [errors, setErrors] = useState<Set<string>>(new Set());
  const [x, setX] = useState(props.center[0].toString());
  const [y, setY] = useState(props.center[1].toString());
  const [zoomLevel, setZoomLevel] = useState(props.zoomLevel);

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

  const validateX = useCallback(
    (value: string) => {
      const result = tryParseNumber(value);
      addOrRemoveErrors(["x"], result.err);
    },
    [addOrRemoveErrors]
  );

  const validateY = useCallback(
    (value: string) => {
      const result = tryParseNumber(value);
      addOrRemoveErrors(["y"], result.err);
    },
    [addOrRemoveErrors]
  );

  return (
    <Dialog
      dialogContentProps={{ title: "Go To" }}
      hidden={false}
      onDismiss={props.dismiss}
    >
      <>
        <div
          style={{
            alignItems: "baseline",
            display: "grid",
            gap: "8px",
            gridTemplateColumns: "auto auto",
          }}
        >
          <Label style={{ textAlign: "right" }}>x:</Label>
          <TextField
            onChange={(_, value) => {
              if (value === undefined) return;
              setX(value);
              validateX(value);
            }}
            styles={decimalInputStyles}
            value={x.toString()}
          />
          <Label style={{ textAlign: "right" }}>y:</Label>
          <TextField
            onChange={(_, value) => {
              if (value === undefined) return;
              setY(value);
              validateY(value);
            }}
            styles={decimalInputStyles}
            value={y.toString()}
          />
          <Label style={{ textAlign: "right" }}>Zoom level:</Label>
          <SpinButton
            min={-BASE_ZOOM_LEVEL}
            max={BASE_ZOOM_LEVEL}
            onChange={(_, value) => {
              if (value === undefined) return;
              const zoomLevel = parseFloat(value);
              setZoomLevel(zoomLevel);
            }}
            styles={integerInputStyles}
            value={zoomLevel.toString()}
          />
        </div>

        <DialogFooter>
          <DefaultButton onClick={props.dismiss} text="Cancel" />
          <PrimaryButton
            disabled={errors.size > 0}
            onClick={() => {
              props.goTo(
                [Number.parseFloat(x), Number.parseFloat(y)],
                zoomLevel
              );
              props.dismiss();
            }}
            text="Go"
          />
        </DialogFooter>
      </>
    </Dialog>
  );
};
