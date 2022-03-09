import {
  DefaultButton,
  Dialog,
  DialogFooter,
  Label,
  PrimaryButton,
} from "@fluentui/react";
import { debounce } from "lodash";
import * as React from "react";
import { useCallback, useMemo, useState } from "react";
import { BASE_ZOOM_LEVEL } from "../common/constants";
import { tryParseIntegerInRange, tryParseNumber } from "../common/parse";
import { SendableTextField } from "./SendableTextField";

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
  const [zoomLevel, setZoomLevel] = useState(props.zoomLevel.toString());

  const [xErrorMessage, setXErrorMessage] = useState<string>();
  const [yErrorMessage, setYErrorMessage] = useState<string>();
  const [zoomLevelErrorMessage, setZoomLevelErrorMessage] = useState<string>();

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

  const send = useCallback(() => {
    if (errors.size > 0) return;
    props.goTo(
      [Number.parseFloat(x), Number.parseFloat(y)],
      Number.parseInt(zoomLevel)
    );
    props.dismiss();
  }, [errors, props, x, y, zoomLevel]);

  const validateX = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseNumber(value);
        setXErrorMessage(addOrRemoveErrors(["x"], result.err));
      }, 200),
    [addOrRemoveErrors]
  );

  const validateY = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseNumber(value);
        setYErrorMessage(addOrRemoveErrors(["y"], result.err));
      }, 200),
    [addOrRemoveErrors]
  );

  const validateZoomLevel = useMemo(
    () =>
      debounce((value: string) => {
        const result = tryParseIntegerInRange(
          value,
          -BASE_ZOOM_LEVEL,
          BASE_ZOOM_LEVEL
        );
        setZoomLevelErrorMessage(addOrRemoveErrors(["zoom-level"], result.err));
      }, 200),
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
          <SendableTextField
            errorMessage={xErrorMessage}
            onChange={(_, value) => {
              if (value === undefined) return;
              setX(value);
              validateX(value);
            }}
            onSend={send}
            styles={decimalInputStyles}
            value={x.toString()}
          />
          <Label style={{ textAlign: "right" }}>y:</Label>
          <SendableTextField
            errorMessage={yErrorMessage}
            onChange={(_, value) => {
              if (value === undefined) return;
              setY(value);
              validateY(value);
            }}
            onSend={send}
            styles={decimalInputStyles}
            value={y.toString()}
          />
          <Label style={{ textAlign: "right" }}>Zoom level:</Label>
          <SendableTextField
            errorMessage={zoomLevelErrorMessage}
            onChange={(_, value) => {
              if (value === undefined) return;
              setZoomLevel(value);
              validateZoomLevel(value);
            }}
            onSend={send}
            styles={integerInputStyles}
            value={zoomLevel.toString()}
          />
        </div>

        <DialogFooter>
          <DefaultButton onClick={props.dismiss} text="Cancel" />
          <PrimaryButton disabled={errors.size > 0} onClick={send} text="Go" />
        </DialogFooter>
      </>
    </Dialog>
  );
};
