import { BigNumber as BN } from "bignumber.js";
import { IBigNumber } from "./IBigNumber";

type BigNumber = IBigNumber<BigNumber>;

const BigNumberConstructor = BN.clone();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(BigNumberConstructor.prototype as any).ceil = function () {
  return this.integerValue(BN.ROUND_CEIL);
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(BigNumberConstructor.prototype as any).floor = function () {
  return this.integerValue(BN.ROUND_FLOOR);
};

/**
 * Constructs an instance of {@link BigNumber}.
 */
function bignum(x: number | string | BigNumber): BigNumber {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return BigNumberConstructor(x as any) as any as BigNumber;
}

export { bignum, BigNumber };
