import { bignum, BigNumber } from "../common/bignumber";

BigNumber.config({
  EXPONENTIAL_AT: 5,
  // Division is used for inverting mantissas and transform to pixel coordinates,
  // which do not require much precision.
  DECIMAL_PLACES: 2,
});

const ZERO: BigNumber = bignum(0);
const ONE: BigNumber = bignum(1);

const LABEL_FONT = "14px 'Noto Sans'";
const BACKGROUND_COLOR = "white";
const TICK_LENGTH_PER_SIDE = 3.5;

const AXES_COLOR = "black";
const AXES_COLOR_SECONDARY = "gray";

const MAJOR_GRID_COLOR = "silver";
const MAJOR_GRID_LINE_DASH: number[] = [];

const MINOR_GRID_COLOR = "silver";
const MINOR_GRID_LINE_DASH = [1, 1];

/**
 * Returns true if the numbers are ordered as _a_ ≤ _b_ ≤ _c_.
 */
function ordered(a: number, b: number, c: number): boolean {
  return a <= b && b <= c;
}

/**
 * Returns 0 if the numbers are ordered as _a_ ≤ _b_ ≤ _c_ ≤ _d_;
 * otherwise, returns the number _x_ that satisfies _a_ ≤ _b_ + _x_ ≤ _c_ + _x_ ≤ _d_
 * and has the smallest absolute value.
 *
 * {@link NaN} is returned if such a number does not exist.
 */
function stickyOffset(a: number, b: number, c: number, d: number): number {
  return Math.max(0, a - b) + Math.min(0, d - c);
}

export function loadFonts(): Promise<FontFace[]> {
  return document.fonts.load(LABEL_FONT);
}

export class Bounds {
  constructor(
    readonly xMin: BigNumber,
    readonly xMax: BigNumber,
    readonly yMin: BigNumber,
    readonly yMax: BigNumber
  ) {}
}

export class GridInterval {
  #x?: BigNumber;
  #xInv?: BigNumber;

  constructor(readonly mantissa: BigNumber, readonly exponent: number) {}

  get(): BigNumber {
    return (this.#x ??= this.mantissa.times(ONE.shiftedBy(this.exponent)));
  }

  getInv(): BigNumber {
    return (this.#xInv ??= ONE.div(this.mantissa).times(
      ONE.shiftedBy(-this.exponent)
    ));
  }
}

export interface Transform {
  (x: BigNumber): number;
}

export class AxesRenderer {
  /// The distance between the axes and tick labels.
  readonly labelOffset = 6;
  /// The minimum distance between the map edges and tick labels.
  readonly padding = 6;

  #height: number;
  #width: number;

  constructor(
    readonly ctx: CanvasRenderingContext2D,
    readonly bounds: Bounds,
    readonly tx: Transform,
    readonly ty: Transform,
    readonly mapViewport: DOMRectReadOnly,
    readonly tileViewport: DOMRectReadOnly
  ) {
    this.#height = tileViewport.height;
    this.#width = tileViewport.width;
  }

  #dilate(r: DOMRectReadOnly, radius: number): DOMRectReadOnly {
    return new DOMRectReadOnly(
      r.x - radius,
      r.y - radius,
      r.width + 2 * radius,
      r.height + 2 * radius
    );
  }

  beginDrawAxes() {
    const { ctx } = this;
    ctx.save();
  }

  beginDrawText() {
    const { ctx } = this;
    ctx.save();
    ctx.font = LABEL_FONT;
    ctx.lineJoin = "round";
    ctx.lineWidth = 2;
    ctx.strokeStyle = BACKGROUND_COLOR;
  }

  clearBackground() {
    const { ctx } = this;

    ctx.clearRect(0, 0, this.#width, this.#height);
  }

  drawAxes() {
    const { ctx, tx, ty } = this;

    const cx = tx(ZERO);
    const cy = ty(ZERO);
    ctx.strokeStyle = AXES_COLOR;
    // Do not merge these paths; otherwise, they may not be rendered
    // when the view is too far from the origin.
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, this.#height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(this.#width, cy);
    ctx.stroke();
  }

  drawOriginLabel() {
    const { ctx, tx, ty } = this;

    const cx = tx(ZERO);
    const cy = ty(ZERO);
    const text = "0";
    const m = ctx.measureText(text);
    const args: [string, number, number] = [
      text,
      cx - m.actualBoundingBoxRight - this.labelOffset,
      cy + m.actualBoundingBoxAscent + this.labelOffset,
    ];
    ctx.fillStyle = AXES_COLOR;
    ctx.strokeText(...args);
    ctx.fillText(...args);
  }

  drawXTicks(interval: GridInterval) {
    const { ctx, mapViewport, tileViewport, tx, ty } = this;
    const { xMax, xMin } = this.bounds;

    const cy = ty(ZERO);
    const wy = tileViewport.top + cy;
    const sticky = stickyOffset(mapViewport.top, wy, wy, mapViewport.bottom);
    ctx.strokeStyle = sticky === 0 ? AXES_COLOR : AXES_COLOR_SECONDARY;

    ctx.beginPath();
    const min = xMin.times(interval.getInv()).ceil().minus(ONE);
    const max = xMax.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const x = i.times(interval.get());
      const cx = tx(x);
      const wx = tileViewport.left + cx;
      if (!ordered(mapViewport.left, wx, mapViewport.right)) {
        continue;
      }

      ctx.moveTo(cx, cy - TICK_LENGTH_PER_SIDE + sticky);
      ctx.lineTo(cx, cy + TICK_LENGTH_PER_SIDE + sticky);
    }
    ctx.stroke();
  }

  drawXTickLabels(interval: GridInterval) {
    const { ctx, mapViewport, tileViewport, tx, ty } = this;
    const { xMax, xMin } = this.bounds;

    const cy = ty(ZERO);
    const wy = tileViewport.top + cy;
    ctx.fillStyle = ordered(mapViewport.top, wy, mapViewport.bottom)
      ? AXES_COLOR
      : AXES_COLOR_SECONDARY;

    const min = xMin.times(interval.getInv()).ceil().minus(ONE);
    const max = xMax.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const x = i.times(interval.get());
      const cx = tx(x);
      const wx = tileViewport.left + cx;
      if (!ordered(mapViewport.left, wx, mapViewport.right)) {
        continue;
      }

      const text = this.#format(x);
      const m = ctx.measureText(text);
      const args: [string, number, number] = [
        text,
        cx - (m.actualBoundingBoxLeft + m.actualBoundingBoxRight) / 2,
        cy + m.actualBoundingBoxAscent + this.labelOffset,
      ];
      const textBounds = this.#dilate(
        this.#getBoundingRect(...args),
        this.padding
      );
      const args2: [string, number, number] = [
        text,
        args[1] +
          stickyOffset(
            mapViewport.left,
            textBounds.left,
            textBounds.right,
            mapViewport.right
          ),
        args[2] +
          stickyOffset(
            mapViewport.top,
            textBounds.top,
            textBounds.bottom,
            mapViewport.bottom
          ),
      ];
      ctx.strokeText(...args2);
      ctx.fillText(...args2);
    }
  }

  drawYTicks(interval: GridInterval) {
    const { ctx, mapViewport, tileViewport, tx, ty } = this;
    const { yMax, yMin } = this.bounds;

    const cx = tx(ZERO);
    const wx = tileViewport.left + cx;
    const sticky = stickyOffset(mapViewport.left, wx, wx, mapViewport.right);
    ctx.strokeStyle = sticky === 0 ? AXES_COLOR : AXES_COLOR_SECONDARY;

    ctx.beginPath();
    const min = yMin.times(interval.getInv()).ceil().minus(ONE);
    const max = yMax.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const y = i.times(interval.get());
      const cy = ty(y);
      const wy = tileViewport.top + cy;
      if (!ordered(mapViewport.top, wy, mapViewport.bottom)) {
        continue;
      }

      ctx.moveTo(cx - TICK_LENGTH_PER_SIDE + sticky, cy);
      ctx.lineTo(cx + TICK_LENGTH_PER_SIDE + sticky, cy);
    }
    ctx.stroke();
  }

  drawYTickLabels(interval: GridInterval) {
    const { ctx, mapViewport, tileViewport, tx, ty } = this;
    const { yMax, yMin } = this.bounds;

    const cx = tx(ZERO);
    const wx = tileViewport.left + cx;
    ctx.fillStyle = ordered(mapViewport.left, wx, mapViewport.right)
      ? AXES_COLOR
      : AXES_COLOR_SECONDARY;

    const min = yMin.times(interval.getInv()).ceil().minus(ONE);
    const max = yMax.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const y = i.times(interval.get());
      const cy = ty(y);
      const wy = tileViewport.top + cy;
      if (!ordered(mapViewport.top, wy, mapViewport.bottom)) {
        continue;
      }

      const text = this.#format(y);
      const m = ctx.measureText(text);
      const args: [string, number, number] = [
        text,
        cx - m.actualBoundingBoxRight - this.labelOffset,
        cy + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2,
      ];
      const textBounds = this.#dilate(
        this.#getBoundingRect(...args),
        this.padding
      );
      const args2: [string, number, number] = [
        text,
        args[1] +
          stickyOffset(
            mapViewport.left,
            textBounds.left,
            textBounds.right,
            mapViewport.right
          ),
        args[2] +
          stickyOffset(
            mapViewport.top,
            textBounds.top,
            textBounds.bottom,
            mapViewport.bottom
          ),
      ];
      ctx.strokeText(...args2);
      ctx.fillText(...args2);
    }
  }

  endDraw() {
    const { ctx } = this;
    ctx.restore();
  }

  #format(x: BigNumber): string {
    // Replace hyphen-minuses with minus signs.
    return x.toString().replaceAll("-", "−");
  }

  #getBoundingRect(text: string, cx: number, cy: number): DOMRectReadOnly {
    const wx = this.tileViewport.left + cx;
    const wy = this.tileViewport.top + cy;
    const m = this.ctx.measureText(text);
    return new DOMRectReadOnly(
      wx - m.actualBoundingBoxLeft,
      wy - m.actualBoundingBoxAscent,
      m.actualBoundingBoxLeft + m.actualBoundingBoxRight,
      m.actualBoundingBoxAscent + m.actualBoundingBoxDescent
    );
  }
}

export class GridRenderer {
  #height: number;
  #width: number;

  constructor(
    readonly ctx: CanvasRenderingContext2D,
    readonly bounds: Bounds,
    readonly tx: Transform,
    readonly ty: Transform,
    readonly tileViewport: DOMRectReadOnly
  ) {
    this.#height = tileViewport.height;
    this.#width = tileViewport.width;
  }

  beginDrawMajorGrid() {
    const { ctx } = this;
    ctx.save();
    ctx.setLineDash(MAJOR_GRID_LINE_DASH);
    ctx.strokeStyle = MAJOR_GRID_COLOR;
  }

  beginDrawMinorGrid() {
    const { ctx } = this;
    ctx.save();
    ctx.setLineDash(MINOR_GRID_LINE_DASH);
    ctx.strokeStyle = MINOR_GRID_COLOR;
  }

  endDraw() {
    const { ctx } = this;
    ctx.restore();
  }

  drawGrid(interval: GridInterval, skipEveryNthLine: BigNumber = ZERO) {
    const { ctx, tx, ty } = this;
    const { xMax, xMin, yMax, yMin } = this.bounds;

    ctx.beginPath();
    {
      const min = xMin.times(interval.getInv()).ceil().minus(ONE);
      const max = xMax.times(interval.getInv()).floor().plus(ONE);
      for (let i = min; i.lte(max); i = i.plus(ONE)) {
        if (i.mod(skipEveryNthLine).isZero()) continue;
        const x = i.times(interval.get());
        const cx = tx(x);
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, this.#height);
      }
    }
    {
      const min = yMin.times(interval.getInv()).ceil().minus(ONE);
      const max = yMax.times(interval.getInv()).floor().plus(ONE);
      for (let i = min; i.lte(max); i = i.plus(ONE)) {
        if (i.mod(skipEveryNthLine).isZero()) continue;
        const y = i.times(interval.get());
        const cy = ty(y);
        ctx.moveTo(0, cy);
        ctx.lineTo(this.#width, cy);
      }
    }
    ctx.stroke();
  }

  fillBackground() {
    const { ctx } = this;

    ctx.fillStyle = BACKGROUND_COLOR;
    ctx.fillRect(0, 0, this.#width, this.#height);
  }
}
