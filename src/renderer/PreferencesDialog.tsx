import { Dialog, Label, SpinButton, Stack, Text } from "@fluentui/react";
import * as React from "react";
import { Preferences } from "../common/preferences";
import { useSelector } from "./models/app";

export interface PreferencesDialogProps {
  dismiss: () => void;
  save: (prefs: Preferences) => void;
}

export const PreferencesDialog = (
  props: PreferencesDialogProps
): JSX.Element => {
  const prefs = useSelector((s) => s.preferences);
  if (prefs === undefined) return <></>;

  return (
    <Dialog
      dialogContentProps={{
        title: "Preferences",
      }}
      hidden={false}
      maxWidth={"100vw"}
      onDismiss={props.dismiss}
    >
      <div
        style={{
          alignItems: "baseline",
          display: "grid",
          gap: "8px",
          gridTemplateColumns: "auto auto",
        }}
      >
        <Label style={{ textAlign: "right" }}>Maximum CPU usage:</Label>
        <Stack
          horizontal
          style={{ alignItems: "baseline" }}
          tokens={{ childrenGap: "4px" }}
        >
          <SpinButton
            defaultValue={prefs.maxCpuUsage.toString()}
            max={prefs.constants.numberOfCpus}
            min={1}
            onChange={(_, value) => {
              if (value === undefined) return;
              props.save({ ...prefs, maxCpuUsage: Number(value) });
            }}
            step={1}
            styles={{ root: { width: "80px" } }}
          />
          <Text>×100%</Text>
        </Stack>
      </div>
    </Dialog>
  );
};
