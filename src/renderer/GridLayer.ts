import * as L from "leaflet";
import { bignum, BigNumber } from "../common/bignumber";
import { BASE_ZOOM_LEVEL } from "../common/constants";
import {
  AxesRenderer,
  Bounds,
  getTransform,
  GridRenderer,
  loadFonts,
  suggestGridIntervals,
} from "../common/gridRenderer";

BigNumber.config({
  EXPONENTIAL_AT: 5,
  // Division is used for inverting mantissas and transform to pixel coordinates,
  // which do not require much precision.
  DECIMAL_PLACES: 2,
});

const RETINA_SCALE = window.devicePixelRatio;
const TILE_SIZE = 256;
const TILE_EXTENSION = 1;
const EXTENDED_TILE_SIZE = TILE_SIZE + TILE_EXTENSION;
// `image-rendering: pixelated` does not work well with translations close to -0.5px.
const TRANSFORM = "translate(-0.4990234375px, -0.4990234375px)";

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

export class AxesLayer extends L.GridLayer {
  constructor(options?: L.GridLayerOptions) {
    super(options);
  }

  onAdd(map: L.Map): this {
    super.onAdd(map);
    map.on("move", this.#redrawCurrentTiles, this);
    return this;
  }

  onRemove(map: L.Map): this {
    super.onRemove(map);
    map.off("move", this.#redrawCurrentTiles, this);
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

    loadFonts().then(() => {
      const tileRange = this.#getVisibleTileRange();
      this.#drawTile(inner, coords, tileRange);
      done(undefined, outer);
    });

    return outer;
  }

  #drawTile(tile: HTMLCanvasElement, coords: L.Coords, tileRange: L.Bounds) {
    const widthPerTilef = 2 ** (BASE_ZOOM_LEVEL - coords.z);
    const [s0, s1] = sourcePoints(coords, widthPerTilef);
    const [d0, d1] = destinationPoints();
    const tx = getTransform([s0.x, s1.x], [d0.x, d1.x]);
    const ty = getTransform([s0.y, s1.y], [d0.y, d1.y]);

    const widthPerPixel = widthPerTilef / TILE_SIZE;
    const [interval] = suggestGridIntervals(widthPerPixel);

    const ctx = tile.getContext("2d")!;
    ctx.setTransform(RETINA_SCALE, 0, 0, RETINA_SCALE, 0, 0);
    const mapViewport = this._map.getContainer().getBoundingClientRect();
    const tileViewport = ctx.canvas.getBoundingClientRect();

    const renderer = new AxesRenderer(
      ctx,
      new Bounds(s0.x, s1.x, s0.y, s1.y),
      EXTENDED_TILE_SIZE,
      EXTENDED_TILE_SIZE,
      tx,
      ty,
      mapViewport,
      tileViewport
    );

    renderer.clearBackground();
    renderer.beginDrawAxes();

    if (
      coords.x === -1 ||
      coords.x === 0 ||
      coords.y === -1 ||
      coords.y === 0
    ) {
      renderer.drawAxes();
    }

    if (
      coords.y === -1 ||
      coords.y === 0 ||
      (tileRange.max!.y <= -1 &&
        (coords.y === tileRange.max!.y || coords.y === tileRange.max!.y - 1)) ||
      (tileRange.min!.y >= 0 &&
        (coords.y === tileRange.min!.y || coords.y === tileRange.min!.y + 1))
    ) {
      renderer.drawXTicks(interval);
    }

    if (
      coords.x === -1 ||
      coords.x === 0 ||
      (tileRange.max!.x <= -1 &&
        (coords.x === tileRange.max!.x || coords.x === tileRange.max!.x - 1)) ||
      (tileRange.min!.x >= 0 &&
        (coords.x === tileRange.min!.x || coords.x === tileRange.min!.x + 1))
    ) {
      renderer.drawYTicks(interval);
    }

    renderer.endDraw();
    renderer.beginDrawText();

    if (coords.x === -1 && coords.y === 0) {
      renderer.drawOriginLabel();
    }

    if (
      coords.y === 0 ||
      (tileRange.max!.y <= 0 &&
        (coords.y === tileRange.max!.y || coords.y === tileRange.max!.y - 1)) ||
      (tileRange.min!.y >= 0 &&
        (coords.y === tileRange.min!.y || coords.y === tileRange.min!.y + 1))
    ) {
      renderer.drawXTickLabels(interval);
    }

    if (
      coords.x === -1 ||
      (tileRange.max!.x <= -1 &&
        (coords.x === tileRange.max!.x || coords.x === tileRange.max!.x - 1)) ||
      (tileRange.min!.x >= -1 &&
        (coords.x === tileRange.min!.x || coords.x === tileRange.min!.x + 1))
    ) {
      renderer.drawYTickLabels(interval);
    }

    renderer.endDraw();
  }

  #getVisibleTileRange(): L.Bounds {
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

  #redrawCurrentTiles() {
    const tileRange = this.#getVisibleTileRange();
    // https://github.com/Leaflet/Leaflet/blob/436430db4203a350601e002c8de6a41fae15a4bf/src/layer/tile/GridLayer.js#L318
    for (const key in this._tiles) {
      const tile = this._tiles[key];
      if (!tile.current || !tile.loaded) {
        continue;
      }
      this.#drawTile(
        tile.el.children[0] as HTMLCanvasElement,
        tile.coords,
        tileRange
      );
    }
  }
}

export class GridLayer extends L.GridLayer {
  #showMajor = true;
  #showMinor = true;

  constructor(options?: L.GridLayerOptions) {
    super(options);
  }

  set showMajorGrid(show: boolean) {
    this.#showMajor = show;
    this.redraw();
  }

  set showMinorGrid(show: boolean) {
    this.#showMinor = show;
    this.redraw();
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
      const [majInterval, minInterval] = suggestGridIntervals(widthPerPixel);

      const ctx = inner.getContext("2d")!;
      ctx.setTransform(RETINA_SCALE, 0, 0, RETINA_SCALE, 0, 0);
      const tileViewport = ctx.canvas.getBoundingClientRect();

      const renderer = new GridRenderer(
        ctx,
        new Bounds(s0.x, s1.x, s0.y, s1.y),
        EXTENDED_TILE_SIZE,
        EXTENDED_TILE_SIZE,
        tx,
        ty
      );

      renderer.fillBackground();

      if (this.#showMinor) {
        renderer.beginDrawMinorGrid();
        renderer.drawXGrid(
          minInterval,
          ...(this.#showMajor
            ? [majInterval.get().idiv(minInterval.get())]
            : [])
        );
        renderer.drawYGrid(
          minInterval,
          ...(this.#showMajor
            ? [majInterval.get().idiv(minInterval.get())]
            : [])
        );
        renderer.endDraw();
      }

      if (this.#showMajor) {
        renderer.beginDrawMajorGrid();
        renderer.drawXGrid(majInterval);
        renderer.drawYGrid(majInterval);
        renderer.endDraw();
      }

      done(undefined, outer);
    }, 0);

    return outer;
  }
}
