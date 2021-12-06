import { Callout, FocusZone, Stack } from "@fluentui/react";
import { debounce } from "lodash";
import * as React from "react";
import { useCallback, useRef, useState } from "react";
import { Bar } from "./Bar";
import { BarIconButton } from "./BarIconButton";

export interface SymbolsButtonProps {
  onSymbolChosen: (symbol: string) => void;
  onSymbolPairChosen: (first: string, second: string) => void;
}

export const SymbolsButton = (props: SymbolsButtonProps): JSX.Element => {
  const buttonRef = useRef<HTMLElement>(null);
  const [showCallout, setShowCallout] = useState(false);

  const setShowCalloutDebounced = useCallback(
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
      {showCallout ? (
        <Callout
          gapSpace={0}
          isBeakVisible={false}
          onDismiss={dismiss}
          target={buttonRef}
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
