import { Icon } from "@fluentui/react";
import * as L from "leaflet";
import "leaflet-easybutton/src/easy-button";
import "leaflet-easybutton/src/easy-button.css";
import "leaflet/dist/leaflet.css";
import * as React from "react";
import { forwardRef, useEffect, useState } from "react";
import * as ReactDOM from "react-dom";
import { useStore } from "react-redux";
import { INITIAL_ZOOM_LEVEL } from "../common/constants";
import { GraphLayer } from "./GraphLayer";
import { AxesLayer, GridLayer } from "./GridLayer";
import { useSelector } from "./models/app";

export interface GraphViewProps {
  grow?: boolean;
}

export const GraphView = forwardRef<HTMLDivElement, GraphViewProps>(
  (props, ref) => {
    const graphs = useSelector((s) => s.graphs);
    const showAxes = useSelector((s) => s.showAxes);
    const showMajorGrid = useSelector((s) => s.showMajorGrid);
    const showMinorGrid = useSelector((s) => s.showMinorGrid);
    const [axesLayer] = useState(new AxesLayer().setZIndex(1));
    const [graphLayers] = useState<Map<string, GraphLayer>>(new Map());
    const [gridLayer] = useState(new GridLayer().setZIndex(0));
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
    }, [graphLayers, graphs, map, store]);

    useEffect(() => {
      map?.addLayer(gridLayer);
    }, [map, gridLayer]);

    useEffect(() => {
      if (map === undefined) return;
      if (showAxes) {
        map.addLayer(axesLayer);
      } else {
        map.removeLayer(axesLayer);
      }
    }, [axesLayer, map, showAxes]);

    useEffect(() => {
      if (map === undefined) return;
      gridLayer.showMajorGrid = showMajorGrid;
    }, [gridLayer, map, showMajorGrid]);

    useEffect(() => {
      if (map === undefined) return;
      gridLayer.showMinorGrid = showMinorGrid;
    }, [gridLayer, map, showMinorGrid]);

    useEffect(() => {
      setMap(
        L.map("map", {
          attributionControl: false,
          crs: L.CRS.Simple,
          fadeAnimation: false,
          inertia: false,
          maxBoundsViscosity: 1,
          wheelDebounceTime: 100,
          zoomControl: false,
        })
      );
    }, []);

    useEffect(() => {
      if (map === undefined) return;

      // We first need to set the view before calling `map.getCenter()`, `getZoom()`, etc.
      resetView();

      map
        .on("move", updateMaxZoom)
        .on("zoom", updateMaxBounds)
        .on("zoomstart", onZoomStart);

      const resizeObserver = new window.ResizeObserver(() => {
        map.invalidateSize();
      });
      resizeObserver.observe(map.getContainer());

      L.control
        .zoom({
          position: "topleft",
          zoomInText: "<div id='zoom-in-button' style='font-size: 16px'></div>",
          zoomInTitle: "Zoom In",
          zoomOutText:
            "<div id='zoom-out-button' style='font-size: 16px'></div>",
          zoomOutTitle: "Zoom Out",
        })
        .addTo(map);
      ReactDOM.render(
        <i className="fa-solid fa-plus"></i>,
        document.getElementById("zoom-in-button")
      );
      ReactDOM.render(
        <i className="fa-solid fa-minus"></i>,
        document.getElementById("zoom-out-button")
      );

      L.easyButton(
        "<div id='reset-view-button' style='font-size: 16px'></div>",
        resetView,
        "Reset view"
      ).addTo(map);
      ReactDOM.render(
        <Icon iconName="HomeSolid" />,
        document.getElementById("reset-view-button")
      );

      function onZoomStart() {
        if (map === undefined) return;
        // Workaround for an issue that the zoom animation does not occur when the map is centered.
        // This seems to happen when both of the levels from and to which the map is zoomed are ≥ 129.
        // Leaflet apparently has nothing to do with this condition,
        // so this could be due to a Chromium's behavior.
        const center = map.getCenter();
        if (center.lat === 0 && center.lng === 0) {
          // Displace the map by a pixel. `map.panBy()` rounds the offset,
          // so we cannot pan by less than a pixel.
          map.panBy(new L.Point(1, 1), { animate: false });
        }
      }

      function resetView() {
        const z = INITIAL_ZOOM_LEVEL;
        map?.setMaxZoom(z).setView([0, 0], z);
        updateMaxBounds();
        updateMaxZoom();
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
