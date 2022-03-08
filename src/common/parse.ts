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
  const val = Number(value);
  if (/^\s*[+-]?\d+\s*$/.test(value) && val >= min && val <= max) {
    return ok(val);
  } else {
    return err(`Value must be an integer between ${min} and ${max}.`);
  }
};

export const tryParseNumber = (value: string): Result<number, string> => {
  const val = Number(value);
  if (
    /^\s*[+-]?(\d+(\.\d*)?|\.\d+)([Ee][+-]?\d+)?\s*$/.test(value) &&
    Number.isFinite(val)
  ) {
    return ok(val);
  } else {
    return err(`Value must be a number.`);
  }
};
