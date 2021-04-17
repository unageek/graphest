import { TextField } from "@fluentui/react";
import * as React from "react";
import { useState } from "react";
import * as ipc from "./ipc";

export interface RelationInputProps {
  grow?: boolean;
  onEnterKeyPressed: () => void;
  onRelationChanged: (relation: string) => void;
  relation: string;
}

export const RelationInput = (props: RelationInputProps): JSX.Element => {
  const [rawRelation, setRawRelation] = useState("y = sin(x)");
  const [hasError, setHasError] = useState(false);

  async function getErrorMessage(relation: string): Promise<string> {
    const { error } = await window.ipcRenderer.invoke<ipc.ValidateRelation>(
      ipc.validateRelation,
      relation
    );
    if (error !== undefined) {
      setHasError(true);
      return "invalid relation";
    } else {
      setHasError(false);
      props.onRelationChanged(relation);
      return "";
    }
  }

  return (
    <TextField
      borderless={!hasError}
      onChange={(e) => {
        setHasError(false);
        setRawRelation((e.target as HTMLInputElement).value);
      }}
      onGetErrorMessage={getErrorMessage}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          props.onEnterKeyPressed();
        }
      }}
      styles={(p) => {
        return {
          root: {
            flexGrow: props.grow ? 1 : undefined,
          },
          wrapper: {
            height: "100%",
          },
          fieldGroup: {
            height: "100%",
            background: p.focused ? undefined : "transparent",
          },
          field: {
            height: "100%",
            fontSize: "18px",
          },
          errorMessage: {
            display: "none",
          },
        };
      }}
      value={rawRelation}
    />
  );
};
