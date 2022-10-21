import { Dialog, DialogFooter, Label, PrimaryButton } from "@fluentui/react";
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
      dialogContentProps={{ title: "Colors" }}
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
          <Label style={{ textAlign: "right" }}>Paper:</Label>
          <ColorButton
            color={background}
            onColorChanged={(c) => dispatch(setGraphBackground(c))}
          />
          <Label style={{ textAlign: "right" }}>Axes & Grids:</Label>
          <ColorButton
            color={foreground}
            onColorChanged={(c) => dispatch(setGraphForeground(c))}
          />
        </div>

        <DialogFooter>
          <PrimaryButton onClick={props.dismiss} text="OK" />
        </DialogFooter>
      </>
    </Dialog>
  );
};
