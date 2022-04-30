import { Unsubscribe } from "@reduxjs/toolkit";
import * as L from "leaflet";
import { EXTENDED_GRAPH_TILE_SIZE, GRAPH_TILE_SIZE } from "../common/constants";
import * as ipc from "../common/ipc";
import { Graph, setGraphIsProcessing } from "./models/graph";
import { Store } from "./models/store";

declare module "leaflet" {
  interface GridLayer {
    _removeTile(key: string): void;
    _tileReady(coords: L.Coords, err?: Error, tile?: HTMLElement): void;
  }
}

export class GraphLayer extends L.GridLayer {
  #dilationElement: SVGFEMorphologyElement;
  #graph?: Graph;
  #onGraphingStatusChangedBound: ipc.RendererListener<ipc.GraphingStatusChanged>;
  #onTileReadyBound: ipc.RendererListener<ipc.TileReady>;
  #styleElement: HTMLStyleElement;
  #svgElement: SVGSVGElement;
  #unsubscribeFromStore?: Unsubscribe;

  constructor(
    readonly store: Store,
    readonly graphId: string,
    options?: L.GridLayerOptions
  ) {
    super({
      className: `graph-layer-${graphId}`,
      keepBuffer: 0,
      tileSize: GRAPH_TILE_SIZE,
      updateWhenZooming: false,
      ...options,
    });
    this.#onGraphingStatusChangedBound =
      this.#onGraphingStatusChanged.bind(this);
    this.#onTileReadyBound = this.#onTileReady.bind(this);

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("style", "display: none;");
    const filter = document.createElementNS(ns, "filter");
    filter.setAttribute("id", `graph-layer-filter-${graphId}`);
    const dilation = document.createElementNS(ns, "feMorphology");
    dilation.setAttribute("operator", "dilate");
    dilation.setAttribute("radius", "1");
    filter.appendChild(dilation);
    svg.appendChild(filter);

    const style = document.createElement("style");
    style.textContent = `.graph-layer-${graphId} { filter: url(#graph-layer-filter-${graphId}); }`;

    this.#dilationElement = dilation;
    this.#styleElement = style;
    this.#svgElement = svg;
  }

  onAdd(map: L.Map): this {
    super.onAdd(map);
    document.body.appendChild(this.#styleElement);
    document.body.appendChild(this.#svgElement);
    window.ipcRenderer.on<ipc.GraphingStatusChanged>(
      ipc.graphingStatusChanged,
      this.#onGraphingStatusChangedBound
    );
    window.ipcRenderer.on<ipc.TileReady>(ipc.tileReady, this.#onTileReadyBound);
    this.#unsubscribeFromStore = this.store.subscribe(
      this.#onAppStateChanged.bind(this)
    );
    this.#onAppStateChanged();
    return this;
  }

  onRemove(map: L.Map): this {
    super.onRemove(map);
    if (this.#graph) {
      this.#abortGraphing(this.#graph.relId);
    }
    document.body.removeChild(this.#styleElement);
    document.body.removeChild(this.#svgElement);
    window.ipcRenderer.off<ipc.GraphingStatusChanged>(
      ipc.graphingStatusChanged,
      this.#onGraphingStatusChangedBound
    );
    window.ipcRenderer.off<ipc.TileReady>(
      ipc.tileReady,
      this.#onTileReadyBound
    );
    this.#unsubscribeFromStore?.();
    return this;
  }

  _removeTile(key: string): void {
    super._removeTile(key);
    const tileId = key;
    if (this.#graph) {
      this.#abortGraphing(this.#graph.relId, tileId);
    }
  }

  // https://github.com/Leaflet/Leaflet/blob/0f904a515879fcd08f69b7f51799ee7f18f23fd8/src/layer/tile/GridLayer.js#L816-L817
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected createTile(coords: L.Coords, _: L.DoneCallback): HTMLElement {
    const outer = L.DomUtil.create("div", "leaflet-tile") as HTMLDivElement;
    if (!this.#graph) {
      return outer;
    }

    // Use two <div>s to reduce the flicker that occurs when updating the mask image.
    const inner1 = document.createElement("div");
    const inner2 = document.createElement("div");
    inner2.style.visibility = "hidden";
    for (const inner of [inner1, inner2]) {
      inner.style.position = "absolute";
      inner.style.top = "0";
      inner.style.left = "0";
      inner.style.width = GRAPH_TILE_SIZE + "px";
      inner.style.height = GRAPH_TILE_SIZE + "px";
      inner.style.background = this.#graph.color;
      inner.style.webkitMaskPosition =
        "top -0.4990234375px left -0.4990234375px";
      inner.style.webkitMaskSize = EXTENDED_GRAPH_TILE_SIZE + "px";
      outer.appendChild(inner);
    }

    const tileId = this._tileCoordsToKey(coords);
    window.ipcRenderer.invoke<ipc.RequestTile>(
      ipc.requestTile,
      this.#graph.relId,
      tileId,
      coords
    );

    return outer;
  }

  #abortGraphing(relId: string, tileId?: string): void {
    window.ipcRenderer.invoke<ipc.AbortGraphing>(
      ipc.abortGraphing,
      relId,
      tileId
    );
  }

  #onAppStateChanged() {
    const state = this.store.getState();

    const oldGraph = this.#graph;
    this.#graph = state.graphs.byId[this.graphId];
    if (!this.#graph) return;

    if (this.#graph.color !== oldGraph?.color) {
      this.#updateColor();
    }

    if (this.#graph.penSize !== oldGraph?.penSize) {
      this.#updatePenSize();
    }

    if (this.#graph.relId !== oldGraph?.relId) {
      this.#updateRelation(oldGraph?.relId);
    }
  }

  #onGraphingStatusChanged: ipc.RendererListener<ipc.GraphingStatusChanged> = (
    _,
    relId,
    processing
  ) => {
    if (relId === this.#graph?.relId) {
      this.store.dispatch(setGraphIsProcessing(this.graphId, processing));
    }
  };

  #onTileReady: ipc.RendererListener<ipc.TileReady> = (
    _,
    relId,
    tileId,
    url
  ) => {
    if (relId === this.#graph?.relId) {
      this.#updateTile(tileId, url);
    }
  };

  #updateColor() {
    if (!this.#graph) {
      throw new Error("`this.#graph` must not be undefined");
    }

    for (const key in this._tiles) {
      const tile = this._tiles[key];
      for (const inner of tile.el.children) {
        if (inner instanceof HTMLElement) {
          inner.style.background = this.#graph.color;
        }
      }
    }
  }

  #updatePenSize() {
    if (!this.#graph) {
      throw new Error("`this.#graph` must not be undefined");
    }

    const penSize = this.#graph.penSize;
    const radius = Math.min(Math.max((penSize - 1.0) / 2.0, 0), 1);
    this.#dilationElement.setAttribute("radius", radius.toString());
  }

  async #updateRelation(oldRelId?: string) {
    if (oldRelId !== undefined) {
      this.#abortGraphing(oldRelId);
    }

    this.redraw();
  }

  #updateTile(tileId: string, url: string): void {
    const preload = new Image();
    preload.src = url;
    preload.addEventListener(
      "load",
      () => {
        const tile = this._tiles[tileId];
        if (tile !== undefined) {
          const inner1 = tile.el.children[0] as HTMLElement;
          const inner2 = tile.el.children[1] as HTMLElement;
          const [prev, cur] =
            inner1.style.visibility === "hidden"
              ? [inner2, inner1]
              : [inner1, inner2];
          cur.style.webkitMaskImage = `url(${url})`;
          const firstUpdate = prev.style.webkitMaskImage === "";
          if (firstUpdate) {
            prev.style.visibility = "hidden";
            cur.style.visibility = "";
            this._tileReady(tile.coords, undefined, tile.el);
          } else {
            setTimeout(() => {
              prev.style.visibility = "hidden";
              cur.style.visibility = "";
            }, 100);
          }
        }
      },
      { once: true }
    );
  }
}
