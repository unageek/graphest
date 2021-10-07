import * as L from "leaflet";
import { bignum, BigNumber } from "./BigNumber";
import { BASE_ZOOM_LEVEL } from "./constants";

BigNumber.config({
  EXPONENTIAL_AT: 5,
  // Division is used for inverting mantissas and transform to pixel coordinates,
  // which do not require much precision.
  DECIMAL_PLACES: 2,
});

const ZERO: BigNumber = bignum(0);
const ONE: BigNumber = bignum(1);

const RETINA_SCALE = window.devicePixelRatio;
const TILE_SIZE = 256;
const TILE_EXTENSION = 1;
const EXTENDED_TILE_SIZE = TILE_SIZE + TILE_EXTENSION;
// `image-rendering: pixelated` does not work well with translations close to -0.5px.
const TRANSFORM = "translate(-0.4990234375px, -0.4990234375px)";

const LABEL_FONT = "14px 'Noto Sans'";
const BACKGROUND_COLOR = "white";
const AXIS_COLOR = "black";
const OFF_AXIS_COLOR = "gray";
const TICK_LENGTH_PER_SIDE = 3.5;
const GRID_COLOR = "silver";

interface Transform {
  (x: BigNumber): number;
}

class GridInterval {
  private x?: BigNumber;
  private xInv?: BigNumber;

  constructor(private readonly mant: BigNumber, private readonly exp: number) {}

  get(): BigNumber {
    return (this.x ??= this.mant.times(ONE.shiftedBy(this.exp)));
  }

  getInv(): BigNumber {
    return (this.xInv ??= ONE.div(this.mant).times(ONE.shiftedBy(-this.exp)));
  }
}

/**
 * Returns the 1-D affine transformation that maps each source point
 * to the corresponding destination point.
 * @param fromPoints The source points.
 * @param toPoints The destination points.
 */
function getTransform(
  fromPoints: [BigNumber, BigNumber],
  toPoints: [BigNumber, BigNumber]
): Transform {
  const [x0, x1] = fromPoints;
  const [y0, y1] = toPoints;
  const d = x1.minus(x0);
  const a = y1.minus(y0);
  const b = x1.times(y0).minus(x0.times(y1));
  return (x) => {
    return +a.times(x).plus(b).div(d);
  };
}

const mantissas = [1, 2, 5].map(bignum);
/**
 * Returns the major and minor grid intervals.
 * @param widthPerPixel The width of pixels in real coordinates.
 */
function gridIntervals(widthPerPixel: number): [GridInterval, GridInterval] {
  function interval(level: number): GridInterval {
    const e = Math.floor(level / 3);
    const m = mantissas[level - 3 * e];
    return new GridInterval(m, e);
  }

  const maxDensity = 20; // One minor line per 20px at most.
  const e = Math.floor(Math.log10(widthPerPixel * maxDensity)) - 1;
  let level = 3 * e;
  for (;;) {
    const minInterval = interval(level);
    if (+minInterval.get() / widthPerPixel >= maxDensity) {
      return [interval(level + 2), minInterval];
    }
    level++;
  }
}

class Point {
  constructor(readonly x: BigNumber, readonly y: BigNumber) {}
}

const dst0 = new Point(bignum(0.5), bignum(TILE_SIZE + 0.5));
const dst1 = new Point(bignum(TILE_SIZE + 0.5), bignum(0.5));

/**
 * Returns the destination points of the transformation from real coordinates
 * to pixel coordinates relative to the tile.
 *
 * @see {@link sourcePoints}
 */
function destinationPoints(): [Point, Point] {
  return [dst0, dst1];
}

/**
 * Returns the source points of the transformation from real coordinates
 * to pixel coordinates relative to the tile.
 * @param coords The coordinates of the tile.
 * @param widthPerTile The width of tiles at the level in real coordinates.
 *
 * @see {@link destinationPoints}
 */
function sourcePoints(coords: L.Coords, widthPerTile: number): [Point, Point] {
  const w = bignum(widthPerTile);
  return [
    new Point(w.times(bignum(coords.x)), w.times(bignum(-coords.y - 1))),
    new Point(w.times(bignum(coords.x + 1)), w.times(bignum(-coords.y))),
  ];
}

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

export class AxesLayer extends L.GridLayer {
  /// The distance between the axes and tick labels.
  readonly labelOffset = 6;
  /// The minimum distance between the map edges and tick labels.
  readonly padding = 6;

  constructor(options?: L.GridLayerOptions) {
    super(options);
  }

  onAdd(map: L.Map): this {
    super.onAdd(map);
    map.on("move", this.redrawCurrentTiles, this);
    return this;
  }

  onRemove(map: L.Map): this {
    super.onRemove(map);
    map.off("move", this.redrawCurrentTiles, this);
    return this;
  }

  protected createTile(coords: L.Coords, done: L.DoneCallback): HTMLElement {
    const outer = L.DomUtil.create("div", "leaflet-tile") as HTMLDivElement;
    outer.style.overflow = "clip";
    const inner = document.createElement("canvas");
    inner.width = RETINA_SCALE * EXTENDED_TILE_SIZE;
    inner.height = RETINA_SCALE * EXTENDED_TILE_SIZE;
    inner.style.width = EXTENDED_TILE_SIZE + "px";
    inner.style.height = EXTENDED_TILE_SIZE + "px";
    inner.style.transform = TRANSFORM;
    outer.appendChild(inner);

    document.fonts.load(LABEL_FONT).then(() => {
      const tileRange = this.getVisibleTileRange();
      this.drawTile(inner, coords, tileRange);
      done(undefined, outer);
    });

    return outer;
  }

  private dilate(r: DOMRectReadOnly, radius: number): DOMRectReadOnly {
    return new DOMRectReadOnly(
      r.x - radius,
      r.y - radius,
      r.width + 2 * radius,
      r.height + 2 * radius
    );
  }

  private drawAxes(
    ctx: CanvasRenderingContext2D,
    tx: Transform,
    ty: Transform
  ) {
    const cx = tx(ZERO);
    const cy = ty(ZERO);
    ctx.strokeStyle = AXIS_COLOR;

    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, EXTENDED_TILE_SIZE);
    ctx.moveTo(0, cy);
    ctx.lineTo(EXTENDED_TILE_SIZE, cy);
    ctx.stroke();
  }

  private drawOriginLabel(
    ctx: CanvasRenderingContext2D,
    tx: Transform,
    ty: Transform
  ) {
    const cx = tx(ZERO);
    const cy = ty(ZERO);
    const text = "0";
    const m = ctx.measureText(text);
    const args: [string, number, number] = [
      text,
      cx - m.actualBoundingBoxRight - this.labelOffset,
      cy + m.actualBoundingBoxAscent + this.labelOffset,
    ];
    ctx.fillStyle = AXIS_COLOR;
    ctx.strokeText(...args);
    ctx.fillText(...args);
  }

  private drawXTicks(
    ctx: CanvasRenderingContext2D,
    x0: BigNumber,
    x1: BigNumber,
    interval: GridInterval,
    tx: Transform,
    ty: Transform,
    mapViewport: DOMRectReadOnly,
    tileViewport: DOMRectReadOnly
  ) {
    const cy = ty(ZERO);
    const wy = tileViewport.top + cy;
    const sticky = stickyOffset(mapViewport.top, wy, wy, mapViewport.bottom);
    ctx.strokeStyle = sticky === 0 ? AXIS_COLOR : OFF_AXIS_COLOR;

    ctx.beginPath();
    const min = x0.times(interval.getInv()).ceil().minus(ONE);
    const max = x1.times(interval.getInv()).floor().plus(ONE);
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

  private drawXTickLabels(
    ctx: CanvasRenderingContext2D,
    x0: BigNumber,
    x1: BigNumber,
    interval: GridInterval,
    tx: Transform,
    ty: Transform,
    mapViewport: DOMRectReadOnly,
    tileViewport: DOMRectReadOnly
  ) {
    const cy = ty(ZERO);
    const wy = tileViewport.top + cy;
    ctx.fillStyle = ordered(mapViewport.top, wy, mapViewport.bottom)
      ? AXIS_COLOR
      : OFF_AXIS_COLOR;

    const min = x0.times(interval.getInv()).ceil().minus(ONE);
    const max = x1.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const x = i.times(interval.get());
      const cx = tx(x);
      const wx = tileViewport.left + cx;
      if (!ordered(mapViewport.left, wx, mapViewport.right)) {
        continue;
      }

      const text = this.format(x);
      const m = ctx.measureText(text);
      const args: [string, number, number] = [
        text,
        cx - (m.actualBoundingBoxLeft + m.actualBoundingBoxRight) / 2,
        cy + m.actualBoundingBoxAscent + this.labelOffset,
      ];
      const textBounds = this.dilate(
        this.getBoundingRect(ctx, ...args, tileViewport),
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

  private drawYTicks(
    ctx: CanvasRenderingContext2D,
    y0: BigNumber,
    y1: BigNumber,
    interval: GridInterval,
    tx: Transform,
    ty: Transform,
    mapViewport: DOMRectReadOnly,
    tileViewport: DOMRectReadOnly
  ) {
    const cx = tx(ZERO);
    const wx = tileViewport.left + cx;
    const sticky = stickyOffset(mapViewport.left, wx, wx, mapViewport.right);
    ctx.strokeStyle = sticky === 0 ? AXIS_COLOR : OFF_AXIS_COLOR;

    ctx.beginPath();
    const min = y0.times(interval.getInv()).ceil().minus(ONE);
    const max = y1.times(interval.getInv()).floor().plus(ONE);
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

  private drawYTickLabels(
    ctx: CanvasRenderingContext2D,
    y0: BigNumber,
    y1: BigNumber,
    interval: GridInterval,
    tx: Transform,
    ty: Transform,
    mapViewport: DOMRectReadOnly,
    tileViewport: DOMRectReadOnly
  ) {
    const cx = tx(ZERO);
    const wx = tileViewport.left + cx;
    ctx.fillStyle = ordered(mapViewport.left, wx, mapViewport.right)
      ? AXIS_COLOR
      : OFF_AXIS_COLOR;

    const min = y0.times(interval.getInv()).ceil().minus(ONE);
    const max = y1.times(interval.getInv()).floor().plus(ONE);
    for (let i = min; i.lte(max); i = i.plus(ONE)) {
      if (i.isZero()) continue;
      const y = i.times(interval.get());
      const cy = ty(y);
      const wy = tileViewport.top + cy;
      if (!ordered(mapViewport.top, wy, mapViewport.bottom)) {
        continue;
      }

      const text = this.format(y);
      const m = ctx.measureText(text);
      const args: [string, number, number] = [
        text,
        cx - m.actualBoundingBoxRight - this.labelOffset,
        cy + (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2,
      ];
      const textBounds = this.dilate(
        this.getBoundingRect(ctx, ...args, tileViewport),
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

  private drawTile(
    tile: HTMLCanvasElement,
    coords: L.Coords,
    tileRange: L.Bounds
  ) {
    const widthPerTilef = 2 ** (BASE_ZOOM_LEVEL - coords.z);
    const [s0, s1] = sourcePoints(coords, widthPerTilef);
    const [d0, d1] = destinationPoints();
    const tx = getTransform([s0.x, s1.x], [d0.x, d1.x]);
    const ty = getTransform([s0.y, s1.y], [d0.y, d1.y]);

    const widthPerPixel = widthPerTilef / TILE_SIZE;
    const [interval] = gridIntervals(widthPerPixel);

    const ctx = tile.getContext("2d")!;
    ctx.setTransform(RETINA_SCALE, 0, 0, RETINA_SCALE, 0, 0);
    const mapViewport = this._map.getContainer().getBoundingClientRect();
    const tileViewport = ctx.canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, EXTENDED_TILE_SIZE, EXTENDED_TILE_SIZE);

    ctx.lineWidth = 1;

    if (
      coords.x === -1 ||
      coords.x === 0 ||
      coords.y === -1 ||
      coords.y === 0
    ) {
      this.drawAxes(ctx, tx, ty);
    }

    if (
      coords.y === -1 ||
      coords.y === 0 ||
      (tileRange.max!.y <= -1 &&
        (coords.y === tileRange.max!.y || coords.y === tileRange.max!.y - 1)) ||
      (tileRange.min!.y >= 0 &&
        (coords.y === tileRange.min!.y || coords.y === tileRange.min!.y + 1))
    ) {
      this.drawXTicks(
        ctx,
        s0.x,
        s1.x,
        interval,
        tx,
        ty,
        mapViewport,
        tileViewport
      );
    }

    if (
      coords.x === -1 ||
      coords.x === 0 ||
      (tileRange.max!.x <= -1 &&
        (coords.x === tileRange.max!.x || coords.x === tileRange.max!.x - 1)) ||
      (tileRange.min!.x >= 0 &&
        (coords.x === tileRange.min!.x || coords.x === tileRange.min!.x + 1))
    ) {
      this.drawYTicks(
        ctx,
        s0.y,
        s1.y,
        interval,
        tx,
        ty,
        mapViewport,
        tileViewport
      );
    }

    ctx.font = LABEL_FONT;
    ctx.lineJoin = "round";
    ctx.lineWidth = 2;
    ctx.strokeStyle = BACKGROUND_COLOR;

    if (coords.x === -1 && coords.y === 0) {
      this.drawOriginLabel(ctx, tx, ty);
    }

    if (
      coords.y === 0 ||
      (tileRange.max!.y <= 0 &&
        (coords.y === tileRange.max!.y || coords.y === tileRange.max!.y - 1)) ||
      (tileRange.min!.y >= 0 &&
        (coords.y === tileRange.min!.y || coords.y === tileRange.min!.y + 1))
    ) {
      this.drawXTickLabels(
        ctx,
        s0.x,
        s1.x,
        interval,
        tx,
        ty,
        mapViewport,
        tileViewport
      );
    }

    if (
      coords.x === -1 ||
      (tileRange.max!.x <= -1 &&
        (coords.x === tileRange.max!.x || coords.x === tileRange.max!.x - 1)) ||
      (tileRange.min!.x >= -1 &&
        (coords.x === tileRange.min!.x || coords.x === tileRange.min!.x + 1))
    ) {
      this.drawYTickLabels(
        ctx,
        s0.y,
        s1.y,
        interval,
        tx,
        ty,
        mapViewport,
        tileViewport
      );
    }
  }

  private format(x: BigNumber): string {
    // Replace hyphen-minuses with minus signs.
    return x.toString().replaceAll("-", "−");
  }

  private getBoundingRect(
    ctx: CanvasRenderingContext2D,
    text: string,
    cx: number,
    cy: number,
    tileViewport: DOMRectReadOnly
  ): DOMRectReadOnly {
    const wx = tileViewport.left + cx;
    const wy = tileViewport.top + cy;
    const m = ctx.measureText(text);
    return new DOMRectReadOnly(
      wx - m.actualBoundingBoxLeft,
      wy - m.actualBoundingBoxAscent,
      m.actualBoundingBoxLeft + m.actualBoundingBoxRight,
      m.actualBoundingBoxAscent + m.actualBoundingBoxDescent
    );
  }

  private getVisibleTileRange(): L.Bounds {
    const bounds = this._map.getPixelBounds();
    return new L.Bounds(
      new L.Point(
        Math.floor(bounds.min!.x / TILE_SIZE),
        Math.floor(bounds.min!.y / TILE_SIZE)
      ),
      new L.Point(
        Math.ceil((bounds.max!.x - (TILE_SIZE - 1)) / TILE_SIZE),
        Math.ceil((bounds.max!.y - (TILE_SIZE - 1)) / TILE_SIZE)
      )
    );
  }

  private redrawCurrentTiles() {
    const tileRange = this.getVisibleTileRange();
    // https://github.com/Leaflet/Leaflet/blob/436430db4203a350601e002c8de6a41fae15a4bf/src/layer/tile/GridLayer.js#L318
    for (const key in this._tiles) {
      const tile = this._tiles[key];
      if (!tile.current || !tile.loaded) {
        continue;
      }
      this.drawTile(
        tile.el.children[0] as HTMLCanvasElement,
        tile.coords,
        tileRange
      );
    }
  }
}

export class GridLayer extends L.GridLayer {
  constructor(options?: L.GridLayerOptions) {
    super(options);
  }

  protected createTile(coords: L.Coords, done: L.DoneCallback): HTMLElement {
    const outer = L.DomUtil.create("div", "leaflet-tile") as HTMLDivElement;
    outer.style.overflow = "clip";
    const inner = document.createElement("canvas");
    inner.width = RETINA_SCALE * EXTENDED_TILE_SIZE;
    inner.height = RETINA_SCALE * EXTENDED_TILE_SIZE;
    inner.style.width = EXTENDED_TILE_SIZE + "px";
    inner.style.height = EXTENDED_TILE_SIZE + "px";
    inner.style.transform = TRANSFORM;
    outer.appendChild(inner);

    setTimeout(() => {
      const widthPerTilef = 2 ** (BASE_ZOOM_LEVEL - coords.z);
      const [s0, s1] = sourcePoints(coords, widthPerTilef);
      const [d0, d1] = destinationPoints();
      const tx = getTransform([s0.x, s1.x], [d0.x, d1.x]);
      const ty = getTransform([s0.y, s1.y], [d0.y, d1.y]);

      const widthPerPixel = widthPerTilef / TILE_SIZE;
      const [majInterval, minInterval] = gridIntervals(widthPerPixel);

      const ctx = inner.getContext("2d")!;
      ctx.setTransform(RETINA_SCALE, 0, 0, RETINA_SCALE, 0, 0);

      ctx.fillStyle = BACKGROUND_COLOR;
      ctx.fillRect(0, 0, EXTENDED_TILE_SIZE, EXTENDED_TILE_SIZE);

      ctx.strokeStyle = GRID_COLOR;

      ctx.setLineDash([1, 1]);
      this.drawGrid(
        ctx,
        s0.x,
        s0.y,
        s1.x,
        s1.y,
        minInterval,
        tx,
        ty,
        majInterval.get().idiv(minInterval.get())
      );

      ctx.setLineDash([]);
      this.drawGrid(ctx, s0.x, s0.y, s1.x, s1.y, majInterval, tx, ty);

      done(undefined, outer);
    }, 0);

    return outer;
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D,
    x0: BigNumber,
    y0: BigNumber,
    x1: BigNumber,
    y1: BigNumber,
    interval: GridInterval,
    tx: Transform,
    ty: Transform,
    skipEveryNthLine: BigNumber = ZERO
  ) {
    ctx.beginPath();
    {
      const min = x0.times(interval.getInv()).ceil().minus(ONE);
      const max = x1.times(interval.getInv()).floor().plus(ONE);
      for (let i = min; i.lte(max); i = i.plus(ONE)) {
        if (i.mod(skipEveryNthLine).isZero()) continue;
        const x = i.times(interval.get());
        const cx = tx(x);
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, EXTENDED_TILE_SIZE);
      }
    }
    {
      const min = y0.times(interval.getInv()).ceil().minus(ONE);
      const max = y1.times(interval.getInv()).floor().plus(ONE);
      for (let i = min; i.lte(max); i = i.plus(ONE)) {
        if (i.mod(skipEveryNthLine).isZero()) continue;
        const y = i.times(interval.get());
        const cy = ty(y);
        ctx.moveTo(0, cy);
        ctx.lineTo(EXTENDED_TILE_SIZE, cy);
      }
    }
    ctx.stroke();
  }
}
