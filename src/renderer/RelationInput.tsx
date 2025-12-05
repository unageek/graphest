import { debounce } from "lodash";
import {
  ReactNode,
  RefObject,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
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
import { RelationError, RequestRelationResult } from "../common/ipc";
import { Range } from "../common/range";
import {
  NormalizationRules,
  Token,
  TokenKind,
  areBracketsBalanced,
  getDecorations,
  getRightBracket,
  isLeftBracket,
  isRightBracket,
  tokenize,
} from "./relationUtils";

export interface RelationInputActions {
  focus: () => void;
  insertSymbol: (symbol: string) => void;
  insertSymbolPair: (left: string, right: string) => void;
}

export interface RelationInputProps {
  actionsRef?: RefObject<RelationInputActions | null>;
  graphId: string;
  grow?: boolean;
  onEnterKeyPressed: () => void;
  onRelationChanged: (relId: string, rel: string) => void;
  processing: boolean;
  relation: string;
  relationInputByUser: boolean;
  requestRelation: (rel: string) => Promise<RequestRelationResult>;
}

type CustomElement = {
  children: CustomText[];
};

type Decoration = {
  highlightedLeftBracket: boolean;
  highlightedRightBracket: boolean;
  multiplication: boolean;
  syntaxError: boolean;
  validationError: boolean;
  validationErrorAfter: boolean;
};

type CustomText = Decoration & {
  text: string;
};

type CustomEditorProps = {
  tokens: Token[];
};

declare module "slate" {
  interface CustomTypes {
    Editor: S.BaseEditor & ReactEditor & CustomEditorProps;
    Element: CustomElement;
    Text: CustomText;
  }
}

type DecoratedRange = S.Range & Partial<Decoration>;

const shouldInsertBracketPairBefore = (subsequentText: string): boolean => {
  for (const token of tokenize(subsequentText)) {
    if (token.kind === TokenKind.Space) {
      continue;
    }
    if (token.isRightBracket() || token.kind === TokenKind.Comma) {
      return true;
    }
    return false;
  }

  return true; // No tokens.
};

const insertSymbolPair = (editor: S.Editor, left: string, right: string) => {
  // Do not read `editor.selection` after any transformation.
  // See https://github.com/ianstormtaylor/slate/issues/4541
  const sel = editor.selection;
  if (!sel) return;

  const selRef = S.Editor.rangeRef(editor, sel);
  if (!selRef.current) return;

  S.Transforms.insertText(editor, left, {
    at: S.Editor.start(editor, selRef.current),
  });

  const lastSel = selRef.current;
  S.Transforms.insertText(editor, right, {
    at: S.Editor.end(editor, selRef.current),
  });
  // Revert the move of the end cursor by `Transforms.insertText`.
  S.Transforms.select(editor, lastSel);

  selRef.unref();
};

const withRelationEditingExtensions = (editor: S.Editor) => {
  const { deleteBackward, insertText, normalizeNode } = editor;

  editor.deleteBackward = (unit) => {
    let handled = false;

    const sel = editor.selection;
    if (!sel) return;

    if (unit === "character" && S.Range.isCollapsed(sel)) {
      S.Editor.withoutNormalizing(editor, () => {
        const point = S.Editor.point(editor, sel);
        const before = S.Editor.before(editor, point, { unit: "character" });
        const after = S.Editor.after(editor, point, { unit: "character" });
        if (!before || !after) return;

        const prevChar = S.Editor.string(editor, {
          anchor: before,
          focus: point,
        });
        if (!isLeftBracket(prevChar)) return;

        const nextChar = S.Editor.string(editor, {
          anchor: point,
          focus: after,
        });
        if (getRightBracket(prevChar) !== nextChar) return;

        if (!areBracketsBalanced(editor.tokens)) return;

        S.Transforms.delete(editor, {
          at: { anchor: before, focus: after },
        });
        handled = true;
      });
    }

    if (!handled) {
      deleteBackward(unit);
    }
  };

  editor.insertBreak = () => {
    // The default implementation splits the node.

    insertText("\n");
  };

  editor.insertText = (text) => {
    let handled = false;

    const sel = editor.selection;
    if (!sel) return;

    if (S.Range.isCollapsed(sel)) {
      if (isLeftBracket(text)) {
        S.Editor.withoutNormalizing(editor, () => {
          const point = S.Editor.point(editor, sel);
          const end = S.Editor.end(editor, [0]);
          const subsequentText = S.Editor.string(editor, {
            anchor: point,
            focus: end,
          });
          if (
            !shouldInsertBracketPairBefore(subsequentText) ||
            !areBracketsBalanced(editor.tokens)
          ) {
            return;
          }

          insertSymbolPair(editor, text, getRightBracket(text));
          handled = true;
        });
      } else if (isRightBracket(text)) {
        S.Editor.withoutNormalizing(editor, () => {
          const point = S.Editor.point(editor, sel);
          const after = S.Editor.after(editor, point, { unit: "character" });
          if (!after) return;

          const nextChar = S.Editor.string(editor, {
            anchor: point,
            focus: after,
          });
          if (text !== nextChar || !areBracketsBalanced(editor.tokens)) return;

          // Step over the right bracket.
          S.Transforms.select(editor, after);
          handled = true;
        });
      }
    } else {
      if (isLeftBracket(text)) {
        S.Editor.withoutNormalizing(editor, () => {
          const selectedText = S.Editor.string(editor, sel);
          if (!areBracketsBalanced(tokenize(selectedText))) return;

          insertSymbolPair(editor, text, getRightBracket(text));
          handled = true;
        });
      }
    }

    if (!handled) {
      insertText(text);
    }
  };

  editor.insertTextData = (data) => {
    // The default implementation puts each line into a separate node.
    // We treat line breaks as they are.

    const text = data.getData("text/plain");
    if (!text) return false;

    insertText(text);
    return true;
  };

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
  let { children } = props;

  const classNames = [];
  if (leaf.highlightedLeftBracket) {
    classNames.push("highlighted-left-bracket");
  }
  if (leaf.highlightedRightBracket) {
    classNames.push("highlighted-right-bracket");
  }
  if (leaf.multiplication) {
    classNames.push("multiplication-container");
    children = <span className="multiplication">{children}</span>;
  }
  if (leaf.syntaxError) {
    classNames.push("syntax-error");
  }
  if (leaf.validationError) {
    classNames.push("validation-error");
  }
  if (leaf.validationErrorAfter) {
    classNames.push("validation-error-after");
  }

  return (
    <span className={classNames.join(" ")} {...attributes}>
      {children}
    </span>
  );
};

export const RelationInput = (props: RelationInputProps): ReactNode => {
  const {
    onEnterKeyPressed,
    onRelationChanged,
    relation,
    relationInputByUser,
    requestRelation,
  } = props;
  const [editor] = useState<S.Editor>(
    withRelationEditingExtensions(withHistory(withReact(S.createEditor()))),
  );
  const [initialValue] = useState<S.Descendant[]>([
    {
      children: [
        {
          text: props.relation,
          highlightedLeftBracket: false,
          highlightedRightBracket: false,
          multiplication: false,
          syntaxError: false,
          validationError: false,
          validationErrorAfter: false,
        },
      ],
    },
  ]);
  const lastRelation = useRef<string>(relation);
  const lastResult = useRef<RequestRelationResult | null>(null);
  const [validationError, setValidationError] = useState<RelationError>();
  const [showValidationError, setShowValidationError] = useState(false);

  const decorate = useCallback(
    (entry: S.NodeEntry): S.Range[] => {
      const [node, path] = entry;
      const ranges: DecoratedRange[] = [];
      if (!S.Text.isText(node)) return ranges;

      const sel = editor.selection;
      if (!sel) return ranges;

      const decs = getDecorations(
        editor.tokens,
        new Range(S.Range.start(sel).offset, S.Range.end(sel).offset),
      );
      for (const { range: r } of decs.highlightedLeftBrackets) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          highlightedLeftBracket: true,
        });
      }
      for (const { range: r } of decs.highlightedRightBrackets) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          highlightedRightBracket: true,
        });
      }
      for (const { range: r } of decs.multiplications) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          multiplication: true,
        });
      }
      for (const { range: r } of decs.syntaxErrors) {
        ranges.push({
          anchor: { path, offset: r.start },
          focus: { path, offset: r.end },
          syntaxError: true,
        });
      }
      if (validationError && showValidationError) {
        const r = validationError.range;
        const editorEnd = S.Editor.end(editor, [editor.children.length - 1]);
        if (r.start === editorEnd.offset) {
          ranges.push({
            anchor: { path, offset: r.start - 1 },
            focus: editorEnd,
            validationErrorAfter: true,
          });
        } else if (r.start === r.end) {
          ranges.push({
            anchor: { path, offset: r.start },
            focus: editorEnd,
            validationError: true,
          });
        } else {
          ranges.push({
            anchor: { path, offset: r.start },
            focus: { path, offset: r.end },
            validationError: true,
          });
        }
      }

      return ranges;
    },
    [editor, showValidationError, validationError],
  );

  const moveCursorToTheEnd = useCallback(() => {
    const end = S.Editor.end(editor, [editor.children.length - 1]);
    S.Transforms.select(editor, end);
  }, [editor]);

  const updateRelation = useCallback(async () => {
    const rel = S.Node.string(editor);
    if (rel === lastRelation.current && lastResult.current) {
      return lastResult.current;
    }

    const result = await requestRelation(rel);
    onRelationChanged(result.ok ?? "", rel);
    setValidationError(result.err);
    lastRelation.current = rel;
    lastResult.current = result;
    return result;
  }, [editor, onRelationChanged, lastRelation, requestRelation]);

  const updateRelationDebounced = useMemo(
    () =>
      debounce(async (): Promise<RequestRelationResult> => {
        return updateRelation();
      }, 200),
    [updateRelation],
  );

  const updateTokens = useCallback(() => {
    const rel = S.Node.string(editor);
    editor.tokens = [...tokenize(rel)];
  }, [editor]);

  useEffect(() => {
    updateTokens();
    ReactEditor.focus(editor);
    moveCursorToTheEnd();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (relationInputByUser) return;

    S.Editor.withoutNormalizing(editor, () => {
      S.Transforms.delete(editor, {
        at: {
          anchor: S.Editor.start(editor, [0]),
          focus: S.Editor.end(editor, [editor.children.length - 1]),
        },
      });
      S.Transforms.insertText(editor, relation, {
        at: S.Editor.start(editor, [0]),
      });
      moveCursorToTheEnd();
    });
  }, [editor, moveCursorToTheEnd, relation, relationInputByUser]);

  useImperativeHandle(props.actionsRef, () => ({
    focus: () => {
      ReactEditor.focus(editor);
    },
    insertSymbol: (symbol: string) => {
      S.Transforms.insertText(editor, symbol);
    },
    insertSymbolPair: (left: string, right: string) => {
      S.Editor.withoutNormalizing(editor, () => {
        insertSymbolPair(editor, left, right);
      });
    },
  }));

  const slate = useMemo(() => {
    return (
      <Slate
        editor={editor}
        onSelectionChange={() => {
          const sel = editor.selection;
          if (!sel) return;

          // HACK: Do some harmless operations to force Slate to re-render decorations.
          // https://github.com/ianstormtaylor/slate/issues/4483
          editor.withoutNormalizing(() => {
            if (S.Range.isCollapsed(sel)) {
              S.Editor.insertText(editor, " ");
              S.Editor.deleteBackward(editor);
            } else {
              S.Editor.addMark(editor, "foo", "");
              S.Editor.removeMark(editor, "foo");
            }
          });
        }}
        onValueChange={() => {
          setShowValidationError(false);
          updateRelationDebounced();
          updateTokens();
        }}
        initialValue={initialValue}
      >
        <Editable
          className="relation-input"
          decorate={decorate}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              updateRelationDebounced();
              // https://github.com/ianstormtaylor/slate/issues/4721
              updateRelationDebounced.flush()?.then((result) => {
                if (result.ok !== undefined) {
                  onEnterKeyPressed();
                } else {
                  setShowValidationError(true);
                }
              });
              e.preventDefault();
              return;
            }
            if (e.altKey && !e.ctrlKey && !e.metaKey) {
              const symbol = (
                e.shiftKey ? shiftAltKeySymbols : altKeySymbols
              ).get(e.code);
              if (symbol) {
                S.Transforms.insertText(editor, symbol);
                e.preventDefault();
              }
              return;
            }
          }}
          renderLeaf={renderLeaf}
          tabIndex={0}
        />
      </Slate>
    );
  }, [
    decorate,
    editor,
    initialValue,
    onEnterKeyPressed,
    updateRelationDebounced,
    updateTokens,
  ]);

  return (
    <div
      className={`relation-input-outer ${validationError ? "has-error" : ""} ${
        !validationError && props.processing ? "processing" : ""
      }`}
      style={{
        flexGrow: props.grow ? 1 : undefined,
      }}
      title={
        validationError && !showValidationError
          ? "Press the Enter key to see the details of the error."
          : undefined
      }
    >
      {slate}
      {validationError && showValidationError && (
        <div className="relation-input-error-message">
          Error: {validationError.message}
        </div>
      )}
    </div>
  );
};

const altKeySymbols: Map<string, string> = new Map([
  ["KeyA", "α"],
  ["KeyB", "β"],
  ["KeyC", "χ"],
  ["KeyD", "δ"],
  ["KeyE", "ε"],
  ["KeyF", "φ"],
  ["KeyG", "γ"],
  ["KeyH", "η"],
  ["KeyI", "ι"],
  ["KeyK", "κ"],
  ["KeyL", "λ"],
  ["KeyM", "μ"],
  ["KeyN", "ν"],
  ["KeyO", "ο"],
  ["KeyP", "π"],
  ["KeyQ", "θ"],
  ["KeyR", "ρ"],
  ["KeyS", "σ"],
  ["KeyT", "τ"],
  ["KeyU", "υ"],
  ["KeyW", "ω"],
  ["KeyX", "ξ"],
  ["KeyY", "ψ"],
  ["KeyZ", "ζ"],
  ["Comma", "≤"],
  ["Period", "≥"],
  ["Digit1", "¬"],
  ["Digit7", "∧"],
  ["Backslash", "∨"],
]);

const shiftAltKeySymbols: Map<string, string> = new Map([
  ["KeyA", "Α"],
  ["KeyB", "Β"],
  ["KeyC", "Χ"],
  ["KeyD", "Δ"],
  ["KeyE", "Ε"],
  ["KeyF", "Φ"],
  ["KeyG", "Γ"],
  ["KeyH", "Η"],
  ["KeyI", "Ι"],
  ["KeyK", "Κ"],
  ["KeyL", "Λ"],
  ["KeyM", "Μ"],
  ["KeyN", "Ν"],
  ["KeyO", "Ο"],
  ["KeyP", "Π"],
  ["KeyQ", "Θ"],
  ["KeyR", "Ρ"],
  ["KeyS", "Σ"],
  ["KeyT", "Τ"],
  ["KeyU", "Υ"],
  ["KeyW", "Ω"],
  ["KeyX", "Ξ"],
  ["KeyY", "Ψ"],
  ["KeyZ", "Ζ"],
]);
