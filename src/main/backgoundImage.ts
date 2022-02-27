import { createCanvas } from "canvas";
import { parentPort, workerData } from "worker_threads";
import { bignum } from "../common/bignumber";
import { ExportImageOptions } from "../common/exportImage";
import {
  AxesRenderer,
  Bounds,
  getTransform,
  GridRenderer,
  suggestGridIntervals,
} from "../common/gridRenderer";
import { Rect } from "../common/rect";

function makeBackgroundImage(opts: ExportImageOptions): Buffer {
  const { height, width, xMax, xMin, yMax, yMin } = opts;

  const bounds = new Bounds(
    bignum(xMin),
    bignum(xMax),
    bignum(yMin),
    bignum(yMax)
  );
  const tx = getTransform(
    [bounds.xMin, bounds.xMax],
    [bignum(0), bignum(width)]
  );
  const ty = getTransform(
    [bounds.yMin, bounds.yMax],
    [bignum(height), bignum(0)]
  );
  const viewport = new Rect(0, 0, Number(width), Number(height));
  const pixelWidth = +bignum(xMax).minus(bignum(xMin)) / width;
  const pixelHeight = +bignum(yMax).minus(bignum(yMin)) / height;
  const [xMajInterval, xMinInterval] = suggestGridIntervals(+pixelWidth);
  const [yMajInterval, yMinInterval] = suggestGridIntervals(+pixelHeight);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  {
    const renderer = new GridRenderer(ctx, bounds, tx, ty, viewport);
    renderer.fillBackground();
    renderer.beginDrawMinorGrid();
    renderer.drawXGrid(
      xMinInterval,
      xMajInterval.get().idiv(xMinInterval.get())
    );
    renderer.drawYGrid(
      yMinInterval,
      yMajInterval.get().idiv(yMinInterval.get())
    );
    renderer.endDraw();
    renderer.beginDrawMajorGrid();
    renderer.drawXGrid(xMajInterval);
    renderer.drawYGrid(yMajInterval);
    renderer.endDraw();
  }
  {
    const renderer = new AxesRenderer(ctx, bounds, tx, ty, viewport, viewport);
    renderer.beginDrawAxes();
    renderer.drawAxes();
    renderer.drawXTicks(xMajInterval);
    renderer.drawYTicks(yMajInterval);
    renderer.endDraw();
    renderer.beginDrawText();
    renderer.drawOriginLabel();
    renderer.drawXTickLabels(xMajInterval);
    renderer.drawYTickLabels(yMajInterval);
    renderer.endDraw();
  }
  return canvas.toBuffer();
}

const buffer = makeBackgroundImage(workerData as ExportImageOptions);
parentPort?.postMessage(buffer);
