import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  Field,
  Input,
  Label,
} from "@fluentui/react-components";
import { debounce } from "lodash";
import * as React from "react";
import { useCallback, useMemo, useState } from "react";
import { BASE_ZOOM_LEVEL } from "../common/constants";
import { tryParseIntegerInRange, tryParseNumber } from "../common/parse";

export interface GoToDialogProps {
  dismiss: () => void;
  goTo: (center: [number, number], zoomLevel: number) => void;
  center: [number, number];
  zoomLevel: number;
}

const MIN_ZOOM_LEVEL: number = -BASE_ZOOM_LEVEL;
// Leaflet maps cannot be zoomed in to a level greater than 1023.
const MAX_ZOOM_LEVEL: number = 1023 - BASE_ZOOM_LEVEL;

export const GoToDialog = (props: GoToDialogProps): JSX.Element => {
  const { dismiss, goTo } = props;

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

  const submit = useCallback(() => {
    if (errors.size > 0) return;
    goTo(
      [Number.parseFloat(x), Number.parseFloat(y)],
      Number.parseInt(zoomLevel)
    );
    dismiss();
  }, [dismiss, errors, goTo, x, y, zoomLevel]);

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
          MIN_ZOOM_LEVEL,
          MAX_ZOOM_LEVEL
        );
        setZoomLevelErrorMessage(addOrRemoveErrors(["zoom-level"], result.err));
      }, 200),
    [addOrRemoveErrors]
  );

  return (
    <Dialog
      onOpenChange={(_, { open }) => {
        if (!open) {
          props.dismiss();
        }
      }}
      open={true}
    >
      <DialogSurface style={{ width: "fit-content" }}>
        <form onSubmit={submit}>
          <DialogBody>
            <DialogTitle>Go To</DialogTitle>

            <DialogContent
              style={{
                alignItems: "baseline",
                display: "grid",
                gap: "8px",
                gridTemplateColumns: "auto auto",
                margin: "8px auto",
              }}
            >
              <Label style={{ textAlign: "right" }}>x:</Label>
              <Field validationMessage={xErrorMessage}>
                <Input
                  onChange={(_, { value }) => {
                    setX(value);
                    validateX(value);
                  }}
                  style={{ width: "150px" }}
                  value={x.toString()}
                />
              </Field>
              <Label style={{ textAlign: "right" }}>y:</Label>
              <Field validationMessage={yErrorMessage}>
                <Input
                  onChange={(_, { value }) => {
                    setY(value);
                    validateY(value);
                  }}
                  style={{ width: "150px" }}
                  value={y.toString()}
                />
              </Field>
              <Label style={{ textAlign: "right" }}>Zoom level:</Label>
              <Field validationMessage={zoomLevelErrorMessage}>
                <Input
                  onChange={(_, { value }) => {
                    setZoomLevel(value);
                    validateZoomLevel(value);
                  }}
                  style={{ width: "80px" }}
                  value={zoomLevel.toString()}
                />
              </Field>
            </DialogContent>

            <DialogActions>
              <Button onClick={props.dismiss}>Cancel</Button>
              <Button
                appearance="primary"
                disabled={errors.size > 0}
                type="submit"
              >
                Go
              </Button>
            </DialogActions>
          </DialogBody>
        </form>
      </DialogSurface>
    </Dialog>
  );
};
