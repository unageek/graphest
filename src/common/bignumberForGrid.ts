import { BigNumber as BN } from "bignumber.js";
import { IBigNumber } from "./IBigNumber";

type BigNumberForGrid = IBigNumber<BigNumberForGrid>;

const BigNumberForGridConstructor = BN.clone({
  EXPONENTIAL_AT: 5,
  // Division is used for inverting mantissas and transform to pixel coordinates,
  // which do not require much precision.
  DECIMAL_PLACES: 2,
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(BigNumberForGridConstructor.prototype as any).ceil = function () {
  return this.integerValue(BN.ROUND_CEIL);
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
(BigNumberForGridConstructor.prototype as any).floor = function () {
  return this.integerValue(BN.ROUND_FLOOR);
};

/**
 * Constructs an instance of {@link BigNumberForGrid}.
 */
function bignum(x: number | string | BigNumberForGrid): BigNumberForGrid {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return new BigNumberForGridConstructor(x as any) as any as BigNumberForGrid;
}

export { bignum, BigNumberForGrid };
