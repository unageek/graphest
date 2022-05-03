import { BigNumber } from "./bignumber";

const BigNumberForGrid = BigNumber.clone({
  EXPONENTIAL_AT: 5,
  // Division is used for inverting mantissas and transform to pixel coordinates,
  // which do not require much precision.
  DECIMAL_PLACES: 2,
});

BigNumberForGrid.prototype.ceil = BigNumber.prototype.ceil;
BigNumberForGrid.prototype.floor = BigNumber.prototype.floor;

/**
 * A shorthand for calling the constructor of {@link BigNumberForGrid}.
 */
function bignum(x: number | string | BigNumber): BigNumber {
  return new BigNumberForGrid(x);
}

export { bignum, BigNumber };
