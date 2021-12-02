import { Callout, FocusZone, Stack } from "@fluentui/react";
import { useId } from "@fluentui/react-hooks";
import * as React from "react";
import { useState } from "react";
import { Bar } from "./Bar";
import { BarIconButton } from "./BarIconButton";

export interface SymbolsButtonProps {
  onSymbolChosen: (symbol: string) => void;
  onSymbolPairChosen: (first: string, second: string) => void;
}

const DISMISS_DELAY = 200;
const OPEN_DELAY = 200;

export const SymbolsButton = (props: SymbolsButtonProps): JSX.Element => {
  const [dismissTimer, setDismissTimer] = useState<number>();
  const [openTimer, setOpenTimer] = useState<number>();
  const [showCallout, setShowCallout] = useState(false);
  const symbolsButtonId = useId("symbols-button");

  function clearTimers() {
    window.clearTimeout(dismissTimer);
    window.clearTimeout(openTimer);
  }

  function dismiss() {
    setShowCallout(false);
    clearTimers();
  }

  function open() {
    setShowCallout(true);
    clearTimers();
  }

  return (
    <BarIconButton
      iconProps={{ iconName: "Variable" }}
      id={symbolsButtonId}
      onClick={() => open()}
      onMouseEnter={() => {
        clearTimers();
        const timer = window.setTimeout(() => {
          setShowCallout(true);
        }, OPEN_DELAY);
        setOpenTimer(timer);
      }}
      onMouseLeave={() => {
        clearTimers();
        const timer = window.setTimeout(() => {
          setShowCallout(false);
        }, DISMISS_DELAY);
        setDismissTimer(timer);
      }}
      title="Symbols"
    >
      {showCallout ? (
        <Callout
          gapSpace={0}
          isBeakVisible={false}
          onDismiss={() => dismiss()}
          target={`#${symbolsButtonId}`}
        >
          <FocusZone>
            <Stack>
              <Bar>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("π");
                    dismiss();
                  }}
                >
                  π
                </BarIconButton>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("θ");
                    dismiss();
                  }}
                >
                  θ
                </BarIconButton>
              </Bar>
              <Bar>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolPairChosen("⌊", "⌋");
                    dismiss();
                  }}
                >
                  ⌊ ⌋
                </BarIconButton>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolPairChosen("⌈", "⌉");
                    dismiss();
                  }}
                >
                  ⌈ ⌉
                </BarIconButton>
              </Bar>
              <Bar>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("≤");
                    dismiss();
                  }}
                >
                  ≤
                </BarIconButton>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("≥");
                    dismiss();
                  }}
                >
                  ≥
                </BarIconButton>
              </Bar>
              <Bar>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("∧");
                    dismiss();
                  }}
                >
                  ∧
                </BarIconButton>
                <BarIconButton
                  onClick={() => {
                    props.onSymbolChosen("∨");
                    dismiss();
                  }}
                >
                  ∨
                </BarIconButton>
              </Bar>
            </Stack>
          </FocusZone>
        </Callout>
      ) : null}
    </BarIconButton>
  );
};
