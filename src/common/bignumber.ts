import { BigNumber } from "bignumber.js";

declare module "bignumber.js" {
  interface BigNumber {
    ceil(): BigNumber;
    floor(): BigNumber;
  }
}

BigNumber.prototype.ceil = function (): BigNumber {
  return this.integerValue(BigNumber.ROUND_CEIL);
};

BigNumber.prototype.floor = function (): BigNumber {
  return this.integerValue(BigNumber.ROUND_FLOOR);
};

/**
 * A shorthand for calling the constructor of {@link BigNumber}.
 */
function bignum(x: number | string | BigNumber): BigNumber {
  return new BigNumber(x);
}

export { bignum, BigNumber };
