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
import {
  ReactNode,
  SubmitEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  MAX_ZOOM_LEVEL,
  maxCoordinate,
  MIN_ZOOM_LEVEL,
} from "../common/constants";
import {
  tryParseIntegerInRange,
  tryParseNumber,
  tryParseNumberInRange,
} from "../common/parse";

export interface GoToDialogProps {
  dismiss: () => void;
  goTo: (center: [number, number], zoomLevel: number) => void;
  center: [number, number];
  zoomLevel: number;
}

const parseInputs = (x: string, y: string, zoomLevel: string) => {
  const parsedZoomLevel = tryParseIntegerInRange(
    zoomLevel,
    MIN_ZOOM_LEVEL,
    MAX_ZOOM_LEVEL,
  );
  const tryParseCoordinate = (value: string) => {
    if (parsedZoomLevel.ok === undefined) {
      return tryParseNumber(value);
    }
    const max = maxCoordinate(parsedZoomLevel.ok);
    return tryParseNumberInRange(value, -max, max);
  };
  return {
    x: tryParseCoordinate(x),
    y: tryParseCoordinate(y),
    zoomLevel: parsedZoomLevel,
  };
};

export const GoToDialog = (props: GoToDialogProps): ReactNode => {
  const { dismiss, goTo } = props;

  const [x, setX] = useState(props.center[0].toString());
  const [y, setY] = useState(props.center[1].toString());
  const [zoomLevel, setZoomLevel] = useState(props.zoomLevel.toString());

  const [parsed, setParsed] = useState(() => parseInputs(x, y, zoomLevel));
  const hasErrors = Object.values(parsed).some((r) => r.err !== undefined);

  const updateParsed = useMemo(
    () =>
      debounce((x: string, y: string, zoomLevel: string) => {
        setParsed(parseInputs(x, y, zoomLevel));
      }, 200),
    [],
  );

  useEffect(() => {
    updateParsed(x, y, zoomLevel);
  }, [updateParsed, x, y, zoomLevel]);

  const submit = useCallback(
    (e: SubmitEvent) => {
      e.preventDefault();
      const p = parseInputs(x, y, zoomLevel);
      if (
        p.x.ok !== undefined &&
        p.y.ok !== undefined &&
        p.zoomLevel.ok !== undefined
      ) {
        goTo([p.x.ok, p.y.ok], p.zoomLevel.ok);
        dismiss();
      }
    },
    [dismiss, goTo, x, y, zoomLevel],
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
                alignItems: "center",
                display: "grid",
                gap: "8px",
                gridTemplateColumns: "auto auto",
                margin: "8px auto",
              }}
            >
              <Label style={{ textAlign: "right" }}>x:</Label>
              <Field validationMessage={parsed.x.err}>
                <Input
                  onChange={(_, { value }) => setX(value)}
                  style={{ width: "150px" }}
                  value={x.toString()}
                />
              </Field>
              <Label style={{ textAlign: "right" }}>y:</Label>
              <Field validationMessage={parsed.y.err}>
                <Input
                  onChange={(_, { value }) => setY(value)}
                  style={{ width: "150px" }}
                  value={y.toString()}
                />
              </Field>
              <Label style={{ textAlign: "right" }}>Zoom level:</Label>
              <Field validationMessage={parsed.zoomLevel.err}>
                <Input
                  onChange={(_, { value }) => setZoomLevel(value)}
                  style={{ width: "100px" }}
                  value={zoomLevel.toString()}
                />
              </Field>
            </DialogContent>

            <DialogActions>
              <Button onClick={props.dismiss}>Cancel</Button>
              <Button appearance="primary" disabled={hasErrors} type="submit">
                Go
              </Button>
            </DialogActions>
          </DialogBody>
        </form>
      </DialogSurface>
    </Dialog>
  );
};
