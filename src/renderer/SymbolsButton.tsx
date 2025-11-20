import { IButtonStyles } from "@fluentui/react";
import {
  Popover,
  PopoverSurface,
  PopoverTrigger,
  ToolbarButton,
} from "@fluentui/react-components";
import { VariableIcon } from "@fluentui/react-icons-mdl2";
import { debounce } from "lodash";
import * as React from "react";
import { useMemo, useState } from "react";
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
  const [showCallout, setShowCallout] = useState(false);

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
    <Popover open={showCallout} positioning={"below-end"}>
      <PopoverTrigger>
        <ToolbarButton
          icon={<VariableIcon />}
          onClick={open}
          onMouseEnter={openDebounced}
          onMouseLeave={dismissDebounced}
          title="Symbols"
        />
      </PopoverTrigger>
      <PopoverSurface>
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
      </PopoverSurface>
    </Popover>
  );
};
