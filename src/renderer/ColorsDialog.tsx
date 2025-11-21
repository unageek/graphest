import {
  Button,
  Dialog,
  DialogActions,
  DialogBody,
  DialogContent,
  DialogSurface,
  DialogTitle,
  Label,
} from "@fluentui/react-components";
import * as React from "react";
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

export const ColorsDialog = (props: ColorsDialogProps): JSX.Element => {
  const background = useSelector((s) => s.graphBackground);
  const foreground = useSelector((s) => s.graphForeground);
  const dispatch = useDispatch();

  return (
    <Dialog
      open={true}
      onOpenChange={(_, { open }) => {
        if (!open) {
          props.dismiss();
        }
      }}
    >
      <DialogSurface>
        <DialogBody>
          <DialogTitle>Colors</DialogTitle>

          <DialogContent
            style={{
              alignItems: "baseline",
              display: "grid",
              gap: "8px",
              gridTemplateColumns: "auto auto",
            }}
          >
            <Label style={{ textAlign: "right" }}>Background:</Label>
            <ColorButton
              color={background}
              onColorChanged={(c) => dispatch(setGraphBackground(c))}
            />
            <Label style={{ textAlign: "right" }}>Axes and grids:</Label>
            <ColorButton
              color={foreground}
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
