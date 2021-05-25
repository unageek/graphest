import { ITextField, TextField } from "@fluentui/react";
import * as React from "react";
import { RefObject, useImperativeHandle, useRef, useState } from "react";
import * as ipc from "./ipc";

export interface RelationInputActions {
  insertSymbol: (symbol: string) => void;
  insertSymbolPair: (first: string, second: string) => void;
}

export interface RelationInputProps {
  actionsRef?: RefObject<RelationInputActions>;
  grow?: boolean;
  onEnterKeyPressed: () => void;
  onRelationChanged: (relation: string) => void;
  relation: string;
}

export const RelationInput = (props: RelationInputProps): JSX.Element => {
  const [rawRelation, setRawRelation] = useState("y = sin(x)");
  const [hasError, setHasError] = useState(false);
  const textFieldRef = useRef<ITextField>(null);

  useImperativeHandle(props.actionsRef, () => ({
    insertSymbol: (symbol: string) => {
      const start = textFieldRef.current?.selectionStart ?? null;
      const end = textFieldRef.current?.selectionEnd ?? null;
      if (start === null || start === -1 || end === null || end === -1) return;

      const rel = rawRelation;
      setRawRelation(rel.slice(0, start) + symbol + rel.slice(end));

      const newStart = start + symbol.length;
      const newEnd = newStart;
      window.setTimeout(() => {
        textFieldRef.current?.setSelectionRange(newStart, newEnd);
      }, 0);
    },
    insertSymbolPair: (first: string, second: string) => {
      const start = textFieldRef.current?.selectionStart ?? null;
      const end = textFieldRef.current?.selectionEnd ?? null;
      if (start === null || start === -1 || end === null || end === -1) return;

      const rel = rawRelation;
      setRawRelation(
        rel.slice(0, start) +
          first +
          rel.slice(start, end) +
          second +
          rel.slice(end)
      );

      const [newStart, newEnd] =
        start === end
          ? [start + first.length, start + first.length]
          : [start, start + (first.length + (end - start) + second.length)];
      window.setTimeout(() => {
        textFieldRef.current?.setSelectionRange(newStart, newEnd);
      }, 0);
    },
  }));

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
      componentRef={textFieldRef}
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
