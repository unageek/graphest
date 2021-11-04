import { Callout, CommandBarButton, FocusZone, Stack } from "@fluentui/react";
import { useId } from "@fluentui/react-hooks";
import * as React from "react";
import { useState } from "react";
import { Bar } from "./Bar";

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
    <CommandBarButton
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
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("π");
                    dismiss();
                  }}
                >
                  π
                </CommandBarButton>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("θ");
                    dismiss();
                  }}
                >
                  θ
                </CommandBarButton>
              </Bar>
              <Bar>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolPairChosen("⌊", "⌋");
                    dismiss();
                  }}
                >
                  ⌊ ⌋
                </CommandBarButton>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolPairChosen("⌈", "⌉");
                    dismiss();
                  }}
                >
                  ⌈ ⌉
                </CommandBarButton>
              </Bar>
              <Bar>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("≤");
                    dismiss();
                  }}
                >
                  ≤
                </CommandBarButton>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("≥");
                    dismiss();
                  }}
                >
                  ≥
                </CommandBarButton>
              </Bar>
              <Bar>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("∧");
                    dismiss();
                  }}
                >
                  ∧
                </CommandBarButton>
                <CommandBarButton
                  onClick={() => {
                    props.onSymbolChosen("∨");
                    dismiss();
                  }}
                >
                  ∨
                </CommandBarButton>
              </Bar>
            </Stack>
          </FocusZone>
        </Callout>
      ) : null}
    </CommandBarButton>
  );
};
