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
  private graph?: Graph;
  private onGraphingStatusChangedBound: ipc.GraphingStatusChanged["listener"];
  private onTileReadyBound: ipc.TileReady["listener"];
  private unsubscribeFromStore?: Unsubscribe;

  constructor(
    readonly store: Store,
    readonly graphId: string,
    options?: L.GridLayerOptions
  ) {
    super({
      keepBuffer: 0,
      updateWhenZooming: false,
      ...options,
    });
    this.onGraphingStatusChangedBound = this.onGraphingStatusChanged.bind(this);
    this.onTileReadyBound = this.onTileReady.bind(this);
  }

  onAdd(map: L.Map): this {
    super.onAdd(map);
    window.ipcRenderer.on<ipc.GraphingStatusChanged>(
      ipc.graphingStatusChanged,
      this.onGraphingStatusChangedBound
    );
    window.ipcRenderer.on<ipc.TileReady>(ipc.tileReady, this.onTileReadyBound);
    this.unsubscribeFromStore = this.store.subscribe(
      this.onAppStateChanged.bind(this)
    );
    this.onAppStateChanged();
    return this;
  }

  onRemove(map: L.Map): this {
    super.onRemove(map);
    if (this.graph) {
      this.abortGraphing(this.graph.relId);
    }
    window.ipcRenderer.off<ipc.GraphingStatusChanged>(
      ipc.graphingStatusChanged,
      this.onGraphingStatusChangedBound
    );
    window.ipcRenderer.off<ipc.TileReady>(ipc.tileReady, this.onTileReadyBound);
    this.unsubscribeFromStore?.();
    return this;
  }

  _removeTile(key: string): void {
    super._removeTile(key);
    const tileId = key;
    if (this.graph) {
      this.abortGraphing(this.graph.relId, tileId);
    }
  }

  // https://github.com/Leaflet/Leaflet/blob/0f904a515879fcd08f69b7f51799ee7f18f23fd8/src/layer/tile/GridLayer.js#L816-L817
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected createTile(coords: L.Coords, _: L.DoneCallback): HTMLElement {
    const outer = L.DomUtil.create("div", "leaflet-tile") as HTMLDivElement;
    if (!this.graph) {
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
      inner.style.background = this.graph.color;
      inner.style.webkitMaskPosition =
        "top -0.4990234375px left -0.4990234375px";
      inner.style.webkitMaskSize = EXTENDED_GRAPH_TILE_SIZE + "px";
      outer.appendChild(inner);
    }

    const tileId = this._tileCoordsToKey(coords);
    window.ipcRenderer.invoke<ipc.RequestTile>(
      ipc.requestTile,
      this.graph.relId,
      tileId,
      coords
    );

    return outer;
  }

  private abortGraphing(relId: string, tileId?: string): void {
    window.ipcRenderer.invoke<ipc.AbortGraphing>(
      ipc.abortGraphing,
      relId,
      tileId
    );
  }

  private onAppStateChanged() {
    const state = this.store.getState();

    const oldGraph = this.graph;
    this.graph = state.graphs.byId[this.graphId];

    if (this.graph.color !== oldGraph?.color) {
      this.updateColor();
    }

    if (this.graph.relId !== oldGraph?.relId) {
      this.updateRelation(oldGraph?.relId);
    }
  }

  private onGraphingStatusChanged: ipc.GraphingStatusChanged["listener"] = (
    _,
    relId,
    processing
  ) => {
    if (relId === this.graph?.relId) {
      this.store.dispatch(setGraphIsProcessing(this.graphId, processing));
    }
  };

  private onTileReady: ipc.TileReady["listener"] = (_, relId, tileId, url) => {
    if (relId === this.graph?.relId) {
      this.updateTile(tileId, url);
    }
  };

  private updateColor() {
    if (!this.graph) {
      throw new Error("`this.graph` must not be undefined");
    }

    for (const key in this._tiles) {
      const tile = this._tiles[key];
      for (const inner of tile.el.children) {
        if (inner instanceof HTMLElement) {
          inner.style.background = this.graph.color;
        }
      }
    }
  }

  private async updateRelation(oldRelId?: string) {
    if (oldRelId !== undefined) {
      this.abortGraphing(oldRelId);
    }

    this.redraw();
  }

  private updateTile(tileId: string, url: string): void {
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
