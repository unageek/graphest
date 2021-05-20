import { Unsubscribe } from "@reduxjs/toolkit";
import * as L from "leaflet";
import { GRAPH_TILE_SIZE } from "./constants";
import * as ipc from "./ipc";
import { Graph, setGraphIsProcessing } from "./models/graph";
import { Store } from "./models/store";

declare module "leaflet" {
  interface GridLayer {
    _removeTile(key: string): void;
    _tileReady(coords: L.Coords, err?: Error, tile?: HTMLElement): void;
  }
}

export class GraphLayer extends L.GridLayer {
  private lastGraph?: Graph;
  private onGraphingStatusChangedBound: ipc.GraphingStatusChanged["listener"];
  private onTileReadyBound: ipc.TileReady["listener"];
  private relId?: string;
  private unsubscribeFromStore?: Unsubscribe;

  constructor(
    readonly store: Store,
    readonly graphId: string,
    options?: L.GridLayerOptions
  ) {
    super({ keepBuffer: 0, updateWhenZooming: false, ...options });
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
      this.onGraphChanged.bind(this)
    );
    this.onGraphChanged();
    return this;
  }

  onRemove(map: L.Map): this {
    super.onRemove(map);
    this.abortGraphing();
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
    this.abortGraphing(tileId);
  }

  // https://github.com/Leaflet/Leaflet/blob/0f904a515879fcd08f69b7f51799ee7f18f23fd8/src/layer/tile/GridLayer.js#L816-L817
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected createTile(coords: L.Coords, _: L.DoneCallback): HTMLElement {
    const outer = L.DomUtil.create("div", "leaflet-tile") as HTMLDivElement;
    if (this.lastGraph === undefined || this.relId === undefined) {
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
      inner.style.background = this.lastGraph.color;
      outer.appendChild(inner);
    }
    const tileId = this._tileCoordsToKey(coords);
    window.ipcRenderer.invoke<ipc.RequestTile>(
      ipc.requestTile,
      this.relId,
      tileId,
      coords
    );
    return outer;
  }

  private abortGraphing(tileId?: string): void {
    if (this.relId !== undefined) {
      window.ipcRenderer.invoke<ipc.AbortGraphing>(
        ipc.abortGraphing,
        this.relId,
        tileId
      );
    }
  }

  private onGraphChanged() {
    const graph = this.store.getState().graphs.byId[this.graphId];
    if (graph !== undefined && graph !== this.lastGraph) {
      if (graph.color !== this.lastGraph?.color) {
        this.setColor(graph.color);
      }
      if (graph.relation !== this.lastGraph?.relation) {
        this.setRelation(graph.relation);
      }
      this.lastGraph = graph;
    }
  }

  private onGraphingStatusChanged: ipc.GraphingStatusChanged["listener"] = (
    _: Event,
    relId,
    processing
  ) => {
    if (this.relId === relId) {
      this.store.dispatch(setGraphIsProcessing(this.graphId, processing));
    }
  };

  private onTileReady: ipc.TileReady["listener"] = (
    _: Event,
    relId,
    tileId,
    url
  ) => {
    if (this.relId === relId) {
      this.updateTile(tileId, url);
    }
  };

  private setColor(color: string) {
    for (const key in this._tiles) {
      const tile = this._tiles[key];
      for (const inner of tile.el.children) {
        if (inner instanceof HTMLElement) {
          inner.style.background = color;
        }
      }
    }
  }

  private async setRelation(rel: string) {
    const { relId } = await window.ipcRenderer.invoke<ipc.NewRelation>(
      ipc.newRelation,
      rel
    );
    // NB: `abortGraphing` depends on `this.relId`.
    this.abortGraphing();
    this.relId = relId;
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
