export class Range {
  constructor(readonly start: number, readonly end: number) {
    if (start > end) {
      throw new Error("`start` must be less than or equal to `end`");
    }
  }
}
