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

export const tryParseInteger = (value: string): Result<number, string> => {
  const val = Number(value);
  if (/^\s*[+-]?\d+\s*$/.test(value) && Number.isFinite(val)) {
    return ok(val);
  } else {
    return err(`Value must be an integer.`);
  }
};

export const tryParseIntegerInRange = (
  value: string,
  min: number,
  max: number
): Result<number, string> => {
  const result = tryParseInteger(value);
  if (result.ok && result.ok >= min && result.ok <= max) {
    return result;
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

export const tryParseNumberInRange = (
  value: string,
  min: number,
  max: number
): Result<number, string> => {
  const result = tryParseNumber(value);
  if (result.ok && result.ok >= min && result.ok <= max) {
    return result;
  } else {
    return err(`Value must be a number between ${min} and ${max}.`);
  }
};
