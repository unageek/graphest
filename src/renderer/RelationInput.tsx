import { ITextField, TextField } from "@fluentui/react";
import * as React from "react";
import {
  RefObject,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as ipc from "../common/ipc";

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

interface Selection {
  start: number;
  end: number;
}

export const RelationInput = (props: RelationInputProps): JSX.Element => {
  const [rawRelation, setRawRelation] = useState(props.relation);
  const [hasError, setHasError] = useState(false);
  const textFieldRef = useRef<ITextField>(null);

  useEffect(() => {
    if (props.relation !== rawRelation) {
      setRawRelation(props.relation);
    }
  }, [props.relation]);

  useEffect(() => {
    textFieldRef.current?.focus();
  }, [textFieldRef]);

  useImperativeHandle(props.actionsRef, () => ({
    insertSymbol: (symbol: string) => {
      const s = getSelection();
      if (!s) return;

      const r = rawRelation;
      setRawRelation(r.slice(0, s.start) + symbol + r.slice(s.end));

      const start = s.start + symbol.length;
      const end = start;
      setSelectionDeferred({ start, end });
    },
    insertSymbolPair: (first: string, second: string) => {
      const s = getSelection();
      if (!s) return;

      const r = rawRelation;
      setRawRelation(
        r.slice(0, s.start) +
          first +
          r.slice(s.start, s.end) +
          second +
          r.slice(s.end)
      );

      const [start, end] =
        s.start === s.end
          ? [s.start + first.length, s.start + first.length]
          : [
              s.start,
              s.start + (first.length + (s.end - s.start) + second.length),
            ];
      setSelectionDeferred({ start, end });
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

  function getSelection(): Selection | undefined {
    const start = textFieldRef.current?.selectionStart ?? null;
    const end = textFieldRef.current?.selectionEnd ?? null;
    return start === null || start === -1 || end === null || end === -1
      ? undefined
      : { start, end };
  }

  function normalizeRelation(relation: string): string {
    // Replace every hyphen-minus with a minus sign.
    return relation.replaceAll("-", "−");
  }

  function setSelectionDeferred(selection: Selection) {
    window.setTimeout(() => {
      textFieldRef.current?.setSelectionRange(selection.start, selection.end);
    }, 0);
  }

  return (
    <TextField
      borderless={!hasError}
      componentRef={textFieldRef}
      onChange={(_, relation) => {
        if (relation === undefined || relation === rawRelation) {
          return;
        }
        const s = getSelection();
        setHasError(false);
        setRawRelation(normalizeRelation(relation));
        if (s) {
          setSelectionDeferred(s);
        }
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
