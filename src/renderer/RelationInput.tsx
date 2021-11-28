import { debounce } from "lodash";
import * as React from "react";
import {
  RefObject,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
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
import {
  getHighlights,
  normalizeRelation,
  Range,
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
  relation: string;
  relationInputByUser: boolean;
}

type CustomElement = {
  children: CustomText[];
};

type CustomText = {
  text: string;
  error: boolean;
  highlight: boolean;
};

declare module "slate" {
  interface CustomTypes {
    Editor: S.BaseEditor & ReactEditor;
    Element: CustomElement;
    Text: CustomText;
  }
}

type DecorateRange = S.Range & {
  error: boolean;
  highlight: boolean;
};

const withRelationNormalization = (editor: S.Editor) => {
  const { normalizeNode } = editor;
  editor.normalizeNode = (entry) => {
    const [node, path] = entry;

    if (S.Text.isText(node)) {
      const t = normalizeRelation(node.text);
      if (t !== node.text) {
        const lastSel = editor.selection;
        // `path` refers to the entire `Text` node, so this replaces the text.
        S.Transforms.insertText(editor, t, { at: path });
        if (lastSel) {
          S.Transforms.setSelection(editor, lastSel);
        }
      }
      return;
    }

    normalizeNode(entry);
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
  if (leaf.highlight) {
    classNames.push("highlight");
  }

  return (
    <span className={classNames.join(" ")} {...attributes}>
      {children}
    </span>
  );
};

export const RelationInput = (props: RelationInputProps) => {
  const editor = useMemo(
    () => withRelationNormalization(withHistory(withReact(S.createEditor()))),
    []
  );
  const [value, setValue] = useState<S.Descendant[]>([
    {
      children: [
        {
          text: props.relation,
          error: false,
          highlight: false,
        },
      ],
    },
  ]);
  const validate = useCallback(
    debounce(async (rel: string) => {
      const valid = await validateRelation(rel);
      if (valid) {
        props.onRelationChanged(rel);
      }
    }, 200),
    [editor]
  );

  function decorate(entry: S.NodeEntry): S.Range[] {
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
        error: true,
        highlight: false,
      });
    }
    for (const r of decs.highlights) {
      ranges.push({
        anchor: { path, offset: r.start },
        focus: { path, offset: r.end },
        error: false,
        highlight: true,
      });
    }

    return ranges;
  }

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
      onChange={(newValue) => {
        setValue(newValue);
        validate(S.Node.string(editor));
      }}
      value={value}
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
        style={{
          flexGrow: props.grow ? 1 : undefined,
        }}
      />
    </Slate>
  );
};
