import { makeStyles, mergeClasses, tokens } from "@fluentui/react-components";
import { ReactNode } from "react";
import { MenuBar } from "./MenuBar";
import { useSelector } from "./models/app";

export const TitleBar = (): ReactNode => {
  const styles = useStyles();
  const title = useSelector((s) => s.title);

  return (
    <div className={styles.root}>
      <div
        className={mergeClasses(
          styles.content,
          window.platform === "linux" && styles.linuxContent,
        )}
      >
        {window.platform !== "darwin" && <MenuBar />}
        <span
          className={mergeClasses(
            styles.title,
            window.platform === "darwin" && styles.macTitle,
            window.platform === "linux" && styles.linuxTitle,
            window.platform === "win32" && styles.winTitle,
          )}
        >
          {title}
        </span>
      </div>
    </div>
  );
};

const useStyles = makeStyles({
  root: {
    WebkitAppRegion: "drag",
    // Collapses unless the content is extended into the title bar.
    height: "env(titlebar-area-height, 0)",
    overflow: "hidden",
    // Let clicks reach the page to close the open menu, as a native menu bar does.
    "&:has([aria-expanded=true])": {
      WebkitAppRegion: "no-drag",
    },
  },
  // The area not covered by the window controls.
  content: {
    alignItems: "center",
    display: "flex",
    height: "100%",
    marginLeft: "env(titlebar-area-x, 0)",
    width: "env(titlebar-area-width, 0)",
  },
  // Center the title as GNOME and KDE do.
  linuxContent: {
    display: "grid",
    gridTemplateColumns: "1fr minmax(0, max-content) 1fr",
  },
  title: {
    color: tokens.colorNeutralForeground1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  // Same as the native title on macOS 26+.
  macTitle: {
    fontSize: "13px",
    fontWeight: tokens.fontWeightBold,
    paddingLeft: "4px",
  },
  linuxTitle: {
    fontSize: tokens.fontSizeBase300,
    fontWeight: tokens.fontWeightBold,
  },
  winTitle: {
    fontSize: tokens.fontSizeBase200,
    paddingLeft: tokens.spacingHorizontalM,
  },
});
