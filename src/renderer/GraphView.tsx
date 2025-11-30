import { AddFilled, HomeFilled, SubtractFilled } from "@fluentui/react-icons";
import Color from "color";
import * as L from "leaflet";
import { ZoomPanOptions } from "leaflet";
import "leaflet-easybutton/src/easy-button";
import "leaflet-easybutton/src/easy-button.css";
import "leaflet/dist/leaflet.css";
import {
  ComponentProps,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import { useStore } from "react-redux";
import { BASE_ZOOM_LEVEL, INITIAL_ZOOM_LEVEL } from "../common/constants";
import { GraphTheme } from "../common/graphTheme";
import { GraphLayer } from "./GraphLayer";
import { AxesLayer, GridLayer } from "./GridLayer";
import {
  AppState,
  setCenter,
  setResetView,
  setZoomLevel,
  useSelector,
} from "./models/app";

export interface GraphViewProps {
  grow?: boolean;
}

// Make sure that when the window size is odd, such as (599, 799),
// the initial position is exactly (0, 0), and
// you can go to exact coordinates using "Go To" or "Reset view" buttons.

L.Map.include({
  // The function is called as
  //  `setView` → `_resetView` → `_move` → `_getNewPixelOrigin`.
  //
  // Removed `._round()` from the original code:
  //   https://github.com/Leaflet/Leaflet/blob/3b793a3b00ff6a716b84a77f598c8b2bf0ed84dd/src/map/Map.js#L1501-L1504
  _getNewPixelOrigin: function (center: L.Coords, zoom: number) {
    const viewHalf = this.getSize()._divideBy(2);
    return this.project(center, zoom)
      ._subtract(viewHalf)
      ._add(this._getMapPanePos());
  },
});

export const GraphView = (
  props: GraphViewProps & ComponentProps<"div">,
): ReactNode => {
  const graphBackground = useSelector((s) => s.graphBackground);
  const graphForeground = useSelector((s) => s.graphForeground);
  const graphs = useSelector((s) => s.graphs);
  const resetView = useSelector((s) => s.resetView);
  const showAxes = useSelector((s) => s.showAxes);
  const showMajorGrid = useSelector((s) => s.showMajorGrid);
  const showMinorGrid = useSelector((s) => s.showMinorGrid);
  const [axesLayer] = useState(new AxesLayer().setZIndex(1));
  const [graphLayers] = useState<Map<string, GraphLayer>>(new Map());
  const [gridLayer] = useState(new GridLayer().setZIndex(0));
  const [map, setMap] = useState<L.Map | undefined>();
  const store = useStore<AppState>();
  const secondMount = useRef(false);

  const updateMaxBounds = useCallback(() => {
    if (map === undefined) return;
    // To get map coordinates from pixel coordinates, multiply them by `2 ** -zoom`.
    // If the view goes outside this range, Leaflet maps can get stuck.
    const max = Number.MAX_SAFE_INTEGER * 2 ** -map.getZoom();
    const min = -max;
    map.setMaxBounds([
      [min, min],
      [max, max],
    ]);
  }, [map]);

  const updateMaxZoom = useCallback(() => {
    if (map === undefined) return;
    const b = map.getBounds();
    // To get pixel coordinates from map coordinates, multiply them by `2 ** zoom`.
    const maxPixelCoord =
      Math.max(-b.getWest(), b.getEast(), -b.getSouth(), b.getNorth()) *
      2 ** map.getZoom();
    // 52 = ⌊lg(Number.MAX_SAFE_INTEGER)⌋.
    const maxZoom =
      map.getZoom() + Math.max(0, 52 - Math.ceil(Math.log2(maxPixelCoord)));
    // Leaflet maps cannot be zoomed in to a level greater than 1023.
    map.setMaxZoom(Math.min(maxZoom, 1023));
  }, [map]);

  const loadViewFromStore = useCallback(() => {
    if (map === undefined) return;
    const state = store.getState();
    const { center: cc, zoomLevel: zz } = state;
    const x = cc[0] * 2 ** -BASE_ZOOM_LEVEL;
    const y = cc[1] * 2 ** -BASE_ZOOM_LEVEL;
    const z = zz + BASE_ZOOM_LEVEL;
    // Use `{ reset: true }` to set the view exactly.
    map
      .setMaxZoom(Infinity)
      .setView([y, x], z, { reset: true } as ZoomPanOptions);
  }, [map, store]);

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
    gridLayer.showMajorGrid = showMajorGrid;
  }, [gridLayer, showMajorGrid]);

  useEffect(() => {
    gridLayer.showMinorGrid = showMinorGrid;
  }, [gridLayer, showMinorGrid]);

  useEffect(() => {
    const graphTheme: GraphTheme = {
      background: graphBackground,
      foreground: graphForeground,
      secondary: new Color(graphForeground).fade(0.5).toString(),
      tertiary: new Color(graphForeground).fade(0.75).toString(),
    };
    axesLayer.theme = graphTheme;
    gridLayer.theme = graphTheme;
  }, [axesLayer, graphBackground, graphForeground, gridLayer]);

  useEffect(() => {
    if (resetView) {
      loadViewFromStore();
      store.dispatch(setResetView(false));
    }
  }, [loadViewFromStore, map, resetView, store]);

  useEffect(() => {
    // https://stackoverflow.com/a/74609594
    if (process.env.NODE_ENV === "production" || secondMount.current) {
      setMap(
        L.map("map", {
          attributionControl: false,
          crs: L.CRS.Simple,
          fadeAnimation: false,
          inertia: false,
          maxBoundsViscosity: 1,
          wheelDebounceTime: 100,
          zoomControl: false,
        }),
      );
    }
    secondMount.current = true;
  }, []);

  useEffect(() => {
    if (map === undefined) return;

    // We first need to set the view before calling `map.getCenter()`, `getZoom()`, etc.
    loadViewFromStore();
    updateMaxBounds();
    updateMaxZoom();

    map
      .on("moveend", () => {
        updateMaxZoom();
        saveViewToStore();
      })
      .on("zoom", updateMaxBounds)
      .on("zoomend", saveViewToStore)
      .on("zoomstart", onZoomStart);

    const resizeObserver = new window.ResizeObserver(() => {
      map.invalidateSize();
    });
    resizeObserver.observe(map.getContainer());

    L.control
      .zoom({
        position: "topleft",
        zoomInText:
          "<div id='zoom-in-button' style='align-items: center; display: flex; font-size: 20px; height: 100%; justify-content: center;'></div>",
        zoomInTitle: "Zoom in",
        zoomOutText:
          "<div id='zoom-out-button' style='align-items: center; display: flex; font-size: 20px; height: 100%; justify-content: center;'></div>",
        zoomOutTitle: "Zoom out",
      })
      .addTo(map);
    createRoot(document.getElementById("zoom-in-button") as HTMLElement).render(
      <AddFilled />,
    );
    createRoot(
      document.getElementById("zoom-out-button") as HTMLElement,
    ).render(<SubtractFilled />);

    L.easyButton(
      "<div id='reset-view-button' style='align-items: center; display: flex; font-size: 20px; height: 100%; justify-content: center;'></div>",
      () => {
        const zoom = INITIAL_ZOOM_LEVEL;
        // Use `{ reset: true }` to set the view exactly.
        map
          .setMaxZoom(Infinity)
          .setView([0, 0], zoom, { reset: true } as ZoomPanOptions);
      },
      "Reset view",
    ).addTo(map);
    createRoot(
      document.getElementById("reset-view-button") as HTMLElement,
    ).render(<HomeFilled />);

    function onZoomStart() {
      if (map === undefined) return;
      // Workaround for an issue that the zoom animation does not occur when the map is centered.
      // This seems to happen when both of the levels from and to which the map is zoomed are ≥ 129.
      // Leaflet apparently has nothing to do with this condition,
      // so this should be due to Chromium's behavior.
      const center = map.getCenter();
      if (center.lat === 0 && center.lng === 0) {
        // Displace the map by a pixel. `map.panBy()` rounds the offset,
        // so we cannot pan by less than a pixel.
        map.panBy(new L.Point(1, 1), { animate: false });
      }
    }

    function saveViewToStore() {
      if (map === undefined) return;
      const center = map.getCenter();
      const x = center.lng;
      const y = center.lat;
      const z = map.getZoom();
      store.dispatch(
        setCenter([x * 2 ** BASE_ZOOM_LEVEL, y * 2 ** BASE_ZOOM_LEVEL]),
      );
      store.dispatch(setZoomLevel(z - BASE_ZOOM_LEVEL));
    }

    return function cleanup() {
      resizeObserver.disconnect();
      map.remove();
    };
  }, [loadViewFromStore, map, store, updateMaxBounds, updateMaxZoom]);

  return (
    <div
      id="map"
      ref={props.ref}
      style={{
        background: graphBackground,
        flexGrow: props.grow ? 1 : undefined,
      }}
    />
  );
};
