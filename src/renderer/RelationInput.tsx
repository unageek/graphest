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
import {
  BaseEditor,
  createEditor,
  Descendant,
  Editor,
  Node,
  NodeEntry,
  Range,
  Text,
  Transforms,
} from "slate";
import { withHistory } from "slate-history";
import {
  Editable,
  ReactEditor,
  RenderLeafProps,
  Slate,
  withReact,
} from "slate-react";
import {
  normalizeRelation,
  syntaxHighlight,
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

type CustomElement = { children: CustomText[] };
type CustomText = { text: string; error: boolean; highlight: boolean };

declare module "slate" {
  interface CustomTypes {
    Editor: BaseEditor & ReactEditor;
    Element: CustomElement;
    Text: CustomText;
  }
}

type DecorateRange = Range & { error: boolean; highlight: boolean };

const withRelationNormalization = (editor: Editor) => {
  const { normalizeNode } = editor;
  editor.normalizeNode = (entry) => {
    const [node, path] = entry;

    if (Text.isText(node)) {
      const t = normalizeRelation(node.text);
      if (t !== node.text) {
        const lastSel = editor.selection;
        // `path` refers to the entire `Text` node, so this replaces the text.
        Transforms.insertText(editor, t, { at: path });
        if (lastSel) {
          Transforms.setSelection(editor, lastSel);
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
  let { children } = props;

  if (leaf.error) {
    children = <span className="error">{children}</span>;
  }
  if (leaf.highlight) {
    children = <span className="highlight">{children}</span>;
  }

  return <span {...attributes}>{children}</span>;
};

export const RelationInput = (props: RelationInputProps) => {
  const editor = useMemo(
    () => withRelationNormalization(withHistory(withReact(createEditor()))),
    []
  );
  const [value, setValue] = useState<Descendant[]>([
    {
      children: [{ text: props.relation, error: false, highlight: false }],
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

  function decorate(entry: NodeEntry): Range[] {
    const [node, path] = entry;
    const ranges: DecorateRange[] = [];
    if (!Text.isText(node)) return ranges;

    const sel = editor.selection;
    if (!sel) return ranges;

    const rel = Editor.string(editor, path);
    const decs = syntaxHighlight(rel, [
      Range.start(sel).offset,
      Range.end(sel).offset,
    ]);
    for (const pos of decs.error) {
      ranges.push({
        anchor: { path, offset: pos },
        focus: { path, offset: pos + 1 },
        error: true,
        highlight: false,
      });
    }
    for (const pos of decs.highlight) {
      ranges.push({
        anchor: { path, offset: pos },
        focus: { path, offset: pos + 1 },
        error: false,
        highlight: true,
      });
    }

    return ranges;
  }

  function moveCursorToTheEnd() {
    Transforms.select(editor, {
      anchor: Editor.end(editor, [editor.children.length - 1]),
      focus: Editor.end(editor, [editor.children.length - 1]),
    });
  }

  useEffect(() => {
    ReactEditor.focus(editor);
    moveCursorToTheEnd();
  }, [editor]);

  useEffect(() => {
    if (props.relationInputByUser) return;

    Editor.withoutNormalizing(editor, () => {
      Transforms.delete(editor, {
        at: {
          anchor: Editor.start(editor, [0]),
          focus: Editor.end(editor, [editor.children.length - 1]),
        },
      });
      Transforms.insertText(editor, props.relation, {
        at: Editor.start(editor, [0]),
      });
      moveCursorToTheEnd();
    });
  }, [props.relation]);

  useImperativeHandle(props.actionsRef, () => ({
    insertSymbol: (symbol: string) => {
      Transforms.insertText(editor, symbol);
    },
    insertSymbolPair: (first: string, second: string) => {
      Editor.withoutNormalizing(editor, () => {
        // Do not read `editor.selection` after any transformation.
        // See https://github.com/ianstormtaylor/slate/issues/4541
        const sel = editor.selection;
        if (!sel) return;

        const selRef = Editor.rangeRef(editor, sel);
        if (!selRef.current) return;

        Transforms.insertText(editor, first, {
          at: Editor.start(editor, selRef.current),
        });

        const lastSel = selRef.current;
        Transforms.insertText(editor, second, {
          at: Editor.end(editor, lastSel),
        });
        // Revert the move of the end cursor by `Transforms.insertText`.
        Transforms.setSelection(editor, lastSel);

        selRef.unref();
      });
    },
  }));

  return (
    <Slate
      editor={editor}
      onChange={(newValue) => {
        setValue(newValue);
        validate(Node.string(editor));
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
