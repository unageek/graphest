import { FocusTrapCallout, IButtonStyles, useTheme } from "@fluentui/react";
import { debounce } from "lodash";
import * as React from "react";
import { useMemo, useRef, useState } from "react";
import { BarIconButton } from "./BarIconButton";

export interface SymbolsButtonProps {
  onSymbolChosen: (symbol: string) => void;
  onSymbolPairChosen: (first: string, second: string) => void;
}

const SymbolButtonStyles: IButtonStyles = {
  label: {
    fontFamily: "DejaVu Mono",
    wordSpacing: "-0.5ch",
  },
};

export const SymbolsButton = (props: SymbolsButtonProps): JSX.Element => {
  const buttonRef = useRef<HTMLElement>(null);
  const [showCallout, setShowCallout] = useState(false);
  const theme = useTheme();

  const setShowCalloutDebounced = useMemo(
    () =>
      debounce((show: boolean) => {
        setShowCallout(show);
      }, 200),
    []
  );

  function dismiss() {
    setShowCalloutDebounced.cancel();
    setShowCallout(false);
  }

  function dismissDebounced() {
    setShowCalloutDebounced(false);
  }

  function open() {
    setShowCalloutDebounced.cancel();
    setShowCallout(true);
  }

  function openDebounced() {
    setShowCalloutDebounced(true);
  }

  return (
    <BarIconButton
      elementRef={buttonRef}
      iconProps={{ iconName: "Variable" }}
      onClick={open}
      onMouseEnter={openDebounced}
      onMouseLeave={dismissDebounced}
      title="Symbols"
    >
      {showCallout && (
        <FocusTrapCallout
          gapSpace={0}
          isBeakVisible={false}
          onDismiss={dismiss}
          styles={{
            root: {
              boxShadow: theme.effects.elevation8,
            },
          }}
          target={buttonRef}
        >
          <div
            style={{
              display: "grid",
              gridAutoRows: "32px",
              gridTemplateColumns: "repeat(6, 1fr)",
            }}
          >
            <BarIconButton
              onClick={() => {
                props.onSymbolChosen("π");
                dismiss();
              }}
              style={{ gridColumn: "span 3" }}
              styles={SymbolButtonStyles}
              title="pi"
            >
              π
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolChosen("θ");
                dismiss();
              }}
              style={{ gridColumn: "span 3" }}
              styles={SymbolButtonStyles}
              title="theta"
            >
              θ
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolPairChosen("(", ")");
                dismiss();
              }}
              style={{ gridColumn: "span 3" }}
              styles={SymbolButtonStyles}
              title="Parentheses"
            >
              ( )
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolPairChosen("[", "]");
                dismiss();
              }}
              style={{ gridColumn: "span 3" }}
              styles={SymbolButtonStyles}
              title="Square brackets"
            >
              [ ]
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolPairChosen("|", "|");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Absolute value"
            >
              | |
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolPairChosen("⌊", "⌋");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Floor function"
            >
              ⌊ ⌋
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolPairChosen("⌈", "⌉");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Ceiling function"
            >
              ⌈ ⌉
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolChosen("∧");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Logical AND"
            >
              ∧
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolChosen("∨");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Logical OR"
            >
              ∨
            </BarIconButton>
            <BarIconButton
              onClick={() => {
                props.onSymbolChosen("¬");
                dismiss();
              }}
              style={{ gridColumn: "span 2" }}
              styles={SymbolButtonStyles}
              title="Logical NOT"
            >
              ¬
            </BarIconButton>
          </div>
        </FocusTrapCallout>
      )}
    </BarIconButton>
  );
};
