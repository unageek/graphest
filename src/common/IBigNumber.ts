export interface IBigNumber<T> {
  ceil(): T;
  div(n: number | T): T;
  floor(): T;
  idiv(n: number | T): T;
  integerValue(rm: number): T;
  isFinite(): boolean;
  isZero(): boolean;
  lt(n: number | T): boolean;
  lte(n: number | T): boolean;
  minus(n: number | T): T;
  mod(n: number | T): T;
  plus(n: number | T): T;
  pow(n: number | T): T;
  shiftedBy(n: number): T;
  times(n: number | T): T;
}
