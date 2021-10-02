import "leaflet-easybutton/src/easy-button.css";
import "leaflet/dist/leaflet.css";

import { Icon } from "@fluentui/react";
import * as L from "leaflet";
import "leaflet-easybutton/src/easy-button";
import * as React from "react";
import { forwardRef, useEffect, useState } from "react";
import * as ReactDOM from "react-dom";
import { useStore } from "react-redux";
import { BASE_ZOOM_LEVEL } from "./constants";
import { GraphLayer } from "./GraphLayer";
import { GridLayer } from "./GridLayer";
import { useSelector } from "./models/app";

export interface GraphViewProps {
  grow?: boolean;
}

export const GraphView = forwardRef<HTMLDivElement, GraphViewProps>(
  (props, ref) => {
    const graphs = useSelector((s) => s.graphs);
    const showGrid = useSelector((s) => s.showGrid);
    const [graphLayers] = useState<Map<string, GraphLayer>>(new Map());
    const [gridLayer] = useState<GridLayer>(new GridLayer());
    const [map, setMap] = useState<L.Map | undefined>();
    const store = useStore();

    useEffect(() => {
      if (map === undefined) return;
      for (const [id, layer] of graphLayers) {
        if (!(id in graphs.byId)) {
          map.removeLayer(layer);
          graphLayers.delete(id);
        }
      }
      for (const id in graphs.byId) {
        if (!graphLayers.has(id)) {
          const layer = new GraphLayer(store, id);
          map.addLayer(layer);
          graphLayers.set(id, layer);
        }
      }
      graphs.allIds.forEach((id, index) => {
        graphLayers.get(id)?.setZIndex(index + 10);
      });
    }, [map, graphs]);

    useEffect(() => {
      if (map === undefined) return;
      if (showGrid) {
        map.addLayer(gridLayer);
      } else {
        map.removeLayer(gridLayer);
      }
    }, [map, showGrid]);

    useEffect(() => {
      setMap(
        L.map("map", {
          attributionControl: false,
          crs: L.CRS.Simple,
          fadeAnimation: false,
          inertia: false,
          maxBoundsViscosity: 1,
          wheelDebounceTime: 100,
        })
      );
    }, []);

    useEffect(() => {
      if (map === undefined) return;

      map
        .on("move", updateMaxZoom)
        .on("moveend", snapToPixels)
        .on("zoom", updateMaxBounds);

      const resizeObserver = new window.ResizeObserver(() => {
        map.invalidateSize();
      });
      resizeObserver.observe(map.getContainer());

      L.easyButton(
        "<div id='reset-view-button' style='font-size: 16px'></div>",
        resetView,
        "Reset View"
      ).addTo(map);
      ReactDOM.render(
        <Icon iconName="HomeSolid" />,
        document.getElementById("reset-view-button")
      );

      resetView();

      function resetView() {
        const z = BASE_ZOOM_LEVEL - 2;
        map?.setMaxZoom(z).setView([0, 0], z);
        updateMaxBounds();
        updateMaxZoom();
      }

      // Workaround for the issue that the map becomes blurry while/after panning.
      // https://github.com/Leaflet/Leaflet/issues/6069
      function snapToPixels() {
        if (map === undefined) return;
        map.off("moveend", snapToPixels);
        const x = map.getCenter().lat;
        const y = map.getCenter().lng;
        const z = map.getZoom();
        // We need to pan by more than a pixel; otherwise, `setView` does not actually pan the map.
        const TEMP_SHIFT = 2;
        const newXTemp = (x * 2 ** z + TEMP_SHIFT) * 2 ** -z;
        const newYTemp = (y * 2 ** z + TEMP_SHIFT) * 2 ** -z;
        const newX = Math.round(x * 2 ** z) * 2 ** -z;
        const newY = Math.round(y * 2 ** z) * 2 ** -z;
        if (newX !== x || newY !== y) {
          map.setView([newXTemp, newYTemp], z);
          map.setView([newX, newY], z, { animate: false });
        }
        map.on("moveend", snapToPixels);
      }

      function updateMaxBounds() {
        if (map === undefined) return;
        // To get map coordinates from pixel coordinates, multiply them by `2 ** -zoom`.
        // If the view goes outside this range, the Leaflet map can get stuck.
        const max = Number.MAX_SAFE_INTEGER * 2 ** -map.getZoom();
        const min = -max;
        map.setMaxBounds([
          [min, min],
          [max, max],
        ]);
      }

      function updateMaxZoom() {
        if (map === undefined) return;
        const b = map.getBounds();
        // To get pixel coordinates from map coordinates, multiply them by `2 ** zoom`.
        const maxPixelCoord =
          Math.max(-b.getWest(), b.getEast(), -b.getSouth(), b.getNorth()) *
          2 ** map.getZoom();
        // 52 = ⌊lg(Number.MAX_SAFE_INTEGER)⌋.
        const maxZoom =
          map.getZoom() + Math.max(0, 52 - Math.ceil(Math.log2(maxPixelCoord)));
        // The Leaflet map cannot be zoomed in further than 1023.
        map.setMaxZoom(Math.min(maxZoom, 1023));
      }

      return function cleanup() {
        resizeObserver.disconnect();
        map.remove();
      };
    }, [map]);

    return (
      <div
        id="map"
        ref={ref}
        style={{ background: "white", flexGrow: props.grow ? 1 : undefined }}
      />
    );
  }
);
