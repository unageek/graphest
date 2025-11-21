import {
  Button,
  makeStyles,
  Popover,
  PopoverSurface,
  PopoverTrigger,
  ToolbarButton,
} from "@fluentui/react-components";
import { VariableIcon } from "@fluentui/react-icons-mdl2";
import { debounce } from "lodash";
import * as React from "react";
import { useMemo, useState } from "react";

export interface SymbolsButtonProps {
  onSymbolChosen: (symbol: string) => void;
  onSymbolPairChosen: (first: string, second: string) => void;
}

const useStyles = makeStyles({
  symbolButton: {
    fontFamily: "DejaVu Mono",
    minWidth: "unset",
    wordSpacing: "-0.5ch",
  },
});

export const SymbolsButton = (props: SymbolsButtonProps): JSX.Element => {
  const [showCallout, setShowCallout] = useState(false);
  const styles = useStyles();

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
      <PopoverSurface
        onMouseEnter={openDebounced}
        onMouseLeave={dismissDebounced}
        style={{
          padding: "4px",
        }}
      >
        <div
          style={{
            display: "grid",
            gridAutoRows: "32px",
            gridTemplateColumns: "repeat(6, 1fr)",
          }}
        >
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolChosen("π");
              dismiss();
            }}
            style={{ gridColumn: "span 3" }}
            title="pi"
          >
            π
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolChosen("θ");
              dismiss();
            }}
            style={{ gridColumn: "span 3" }}
            title="theta"
          >
            θ
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolPairChosen("(", ")");
              dismiss();
            }}
            style={{ gridColumn: "span 3" }}
            title="Parentheses"
          >
            ( )
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolPairChosen("[", "]");
              dismiss();
            }}
            style={{ gridColumn: "span 3" }}
            title="Square brackets"
          >
            [ ]
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolPairChosen("|", "|");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Absolute value"
          >
            | |
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolPairChosen("⌊", "⌋");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Floor function"
          >
            ⌊ ⌋
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolPairChosen("⌈", "⌉");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Ceiling function"
          >
            ⌈ ⌉
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolChosen("∧");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Logical AND"
          >
            ∧
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolChosen("∨");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Logical OR"
          >
            ∨
          </Button>
          <Button
            appearance="subtle"
            className={styles.symbolButton}
            onClick={() => {
              props.onSymbolChosen("¬");
              dismiss();
            }}
            style={{ gridColumn: "span 2" }}
            title="Logical NOT"
          >
            ¬
          </Button>
        </div>
      </PopoverSurface>
    </Popover>
  );
};
