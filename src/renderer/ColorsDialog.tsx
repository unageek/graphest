import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  Label,
  useId,
} from "@fluentui/react-components";
import { ReactNode } from "react";
import { useDispatch } from "react-redux";
import { ColorButton } from "./ColorButton";
import {
  setGraphBackground,
  setGraphForeground,
  useSelector,
} from "./models/app";

export interface ColorsDialogProps {
  dismiss: () => void;
}

export const ColorsDialog = (props: ColorsDialogProps): ReactNode => {
  const dispatch = useDispatch();
  const background = useSelector((s) => s.graphBackground);
  const foreground = useSelector((s) => s.graphForeground);
  const backgroundId = useId();
  const foregroundId = useId();

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
        <DialogBody>
          <DialogTitle>Colors</DialogTitle>

          <DialogContent
            style={{
              alignItems: "center",
              display: "grid",
              gap: "8px",
              gridTemplateColumns: "auto auto",
              margin: "8px auto",
            }}
          >
            <Label htmlFor={backgroundId} style={{ textAlign: "right" }}>
              Background:
            </Label>
            <ColorButton
              color={background}
              id={backgroundId}
              onColorChanged={(c) => dispatch(setGraphBackground(c))}
            />
            <Label htmlFor={foregroundId} style={{ textAlign: "right" }}>
              Axes and grids:
            </Label>
            <ColorButton
              color={foreground}
              id={foregroundId}
              onColorChanged={(c) => dispatch(setGraphForeground(c))}
            />
          </DialogContent>

          <DialogActions>
            <Button appearance="primary" onClick={props.dismiss}>
              OK
            </Button>
          </DialogActions>
        </DialogBody>
      </DialogSurface>
    </Dialog>
  );
};
