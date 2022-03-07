import { bignum, BigNumber } from "./bignumber";
import { err, ok, Result } from "./result";

export const tryParseBignum = (value: string): Result<BigNumber, string> => {
  const val = bignum(value);
  if (val.isFinite()) {
    return ok(val);
  } else {
    return err("Value must be a number.");
  }
};

export const tryParseIntegerInRange = (
  value: string,
  min: number,
  max: number
): Result<number, string> => {
  const val = Number.parseInt(value);
  if (val >= min && val <= max) {
    return ok(val);
  } else {
    return err(`Value must be an integer between 1 and ${max}.`);
  }
};

export const tryParseNumber = (value: string): Result<number, string> => {
  const val = Number.parseFloat(value);
  if (Number.isFinite(val)) {
    return ok(val);
  } else {
    return err(`Value must be a number.`);
  }
};
