import { debounce } from "lodash";
import * as React from "react";
import {
  RefObject,
  useCallback,
  useEffect,
  useImperativeHandle,
  useState,
} from "react";
import * as S from "slate";
import { withHistory } from "slate-history";
import {
  Editable,
  ReactEditor,
  RenderLeafProps,
  Slate,
  withReact,
} from "slate-react";
import { Range } from "../common/range";
import { ValidationResult } from "../common/validationResult";
import {
  getHighlights,
  NormalizationRules,
  validateRelation,
} from "./relationUtils";

export interface RelationInputActions {
  insertSymbol: (symbol: string) => void;
  insertSymbolPair: (first: string, second: string) => void;
}

export interface RelationInputProps {
  actionsRef?: RefObject<RelationInputActions>;
  grow?: boolean;
  onEnterKeyPressed: () => void;
  onRelationChanged: (relation: string) => void;
  processing: boolean;
  relation: string;
  relationInputByUser: boolean;
}

type CustomElement = {
  children: CustomText[];
};

type CustomText = {
  text: string;
  error: boolean;
  errorAfter: boolean;
  highlightLeft: boolean;
  highlightRight: boolean;
  syntaxError: boolean;
};

declare module "slate" {
  interface CustomTypes {
    Editor: S.BaseEditor & ReactEditor;
    Element: CustomElement;
    Text: CustomText;
  }
}

type DecorateRange = S.Range & {
  error?: boolean;
  errorAfter?: boolean;
  highlightLeft?: boolean;
  highlightRight?: boolean;
  syntaxError?: boolean;
};

const withRelationNormalization = (editor: S.Editor) => {
  const { normalizeNode } = editor;
  editor.normalizeNode = (entry) => {
    const [node, path] = entry;

    if (S.Text.isText(node)) {
      let text = node.text;
      for (const [pat, replace] of NormalizationRules) {
        for (;;) {
          const offset = text.search(pat);
          if (offset === -1) break;

          S.Transforms.delete(editor, {
            at: {
              anchor: { path, offset },
              focus: { path, offset: offset + pat.length },
            },
          });
          S.Transforms.insertText(editor, replace, {
            at: { path, offset },
          });

          text =
            text.slice(0, offset) + replace + text.slice(offset + pat.length);
        }
      }
    } else {
      normalizeNode(entry);
    }
  };

  return editor;
};

const renderLeaf = (props: RenderLeafProps) => {
  const { attributes, leaf } = props;
  const { children } = props;

  const classNames = [];
  if (leaf.error) {
    classNames.push("error");
  }
  if (leaf.errorAfter) {
    classNames.push("error-after");
  }
  if (leaf.highlightLeft) {
    classNames.push("highlight-left");
  }
  if (leaf.highlightRight) {
    classNames.push("highlight-right");
  }
  if (leaf.syntaxError) {
    classNames.push("syntax-error");
  }

  return (
    <span className={classNames.join(" ")} {...attributes}>
      {children}
    </span>
  );
};

export const RelationInput = (props: RelationInputProps) => {
  const [editor] = useState(
    withRelationNormalization(withHistory(withReact(S.createEditor())))
  );

  const [validationResult, setValidationResult] =
    useState<ValidationResult>(null);

  const [value, setValue] = useState<S.Descendant[]>([
    {
      children: [
        {
          text: props.relation,
          error: false,
          errorAfter: false,
          highlightLeft: false,
          highlightRight: false,
          syntaxError: false,
        },
      ],
    },
  ]);

  const validate = useCallback(
    debounce(async (rel: string) => {
      const result = await validateRelation(rel);
      if (result === null) {
        props.onRelationChanged(rel);
      }
      setValidationResult(result);
    }, 200),
    []
  );

  const decorate = useCallback(
    (entry: S.NodeEntry): S.Range[] => {
      const [node, path] = entry;
      const ranges: DecorateRange[] = [];
      if (!S.Text.isText(node)) return ranges;

      const sel = editor.selection;
      if (!sel) return ranges;

      const rel = S.Editor.string(editor, path);
      const decs = getHighlights(
        rel,
        new Range(S.Range.start(sel).offset, S.Range.end(sel).offset)
      );
      for (const r of decs.errors) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          syntaxError: true,
        });
      }
      for (const r of decs.highlightsLeft) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          highlightLeft: true,
        });
      }
      for (const r of decs.highlightsRight) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          highlightRight: true,
        });
      }
      if (validationResult !== null) {
        const r = validationResult.range;
        const editorEnd = S.Editor.end(editor, [editor.children.length - 1]);
        if (r.start === editorEnd.offset) {
          ranges.push({
            anchor: { path, offset: r.start - 1 },
            focus: editorEnd,
            errorAfter: true,
          });
        } else if (r.start === r.end) {
          ranges.push({
            anchor: { path, offset: r.start },
            focus: editorEnd,
            error: true,
          });
        } else {
          ranges.push({
            anchor: { path, offset: r.start },
            focus: { path, offset: r.end },
            error: true,
          });
        }
      }

      return ranges;
    },
    [validationResult]
  );

  function moveCursorToTheEnd() {
    S.Transforms.select(editor, {
      anchor: S.Editor.end(editor, [editor.children.length - 1]),
      focus: S.Editor.end(editor, [editor.children.length - 1]),
    });
  }

  useEffect(() => {
    ReactEditor.focus(editor);
    moveCursorToTheEnd();
  }, [editor]);

  useEffect(() => {
    if (props.relationInputByUser) return;

    S.Editor.withoutNormalizing(editor, () => {
      S.Transforms.delete(editor, {
        at: {
          anchor: S.Editor.start(editor, [0]),
          focus: S.Editor.end(editor, [editor.children.length - 1]),
        },
      });
      S.Transforms.insertText(editor, props.relation, {
        at: S.Editor.start(editor, [0]),
      });
      moveCursorToTheEnd();
    });
  }, [props.relation]);

  useImperativeHandle(props.actionsRef, () => ({
    insertSymbol: (symbol: string) => {
      S.Transforms.insertText(editor, symbol);
    },
    insertSymbolPair: (first: string, second: string) => {
      S.Editor.withoutNormalizing(editor, () => {
        // Do not read `editor.selection` after any transformation.
        // See https://github.com/ianstormtaylor/slate/issues/4541
        const sel = editor.selection;
        if (!sel) return;

        const selRef = S.Editor.rangeRef(editor, sel);
        if (!selRef.current) return;

        S.Transforms.insertText(editor, first, {
          at: S.Editor.start(editor, selRef.current),
        });

        const lastSel = selRef.current;
        S.Transforms.insertText(editor, second, {
          at: S.Editor.end(editor, lastSel),
        });
        // Revert the move of the end cursor by `Transforms.insertText`.
        S.Transforms.setSelection(editor, lastSel);

        selRef.unref();
      });
    },
  }));

  return (
    <Slate
      editor={editor}
      onChange={(value) => {
        // https://github.com/ianstormtaylor/slate/issues/4687#issuecomment-977911063
        if (editor.operations.some((op) => op.type !== "set_selection")) {
          setValue(value);
          setValidationResult(null);
          validate(S.Node.string(editor));
        }
      }}
      value={value}
    >
      <div
        className={`relation-input-outer ${
          props.processing ? "processing" : ""
        }`}
        style={{
          flexGrow: props.grow ? 1 : undefined,
        }}
      >
        <Editable
          className="relation-input"
          decorate={decorate}
          onKeyDown={(e: KeyboardEvent) => {
            if (e.key === "Enter") {
              props.onEnterKeyPressed();
              e.preventDefault();
            }
          }}
          renderLeaf={renderLeaf}
        />
        {validationResult && (
          <div className="relation-input-error-message">
            Error: {validationResult?.message}
          </div>
        )}
      </div>
    </Slate>
  );
};
