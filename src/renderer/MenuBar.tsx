import {
  Button,
  makeStyles,
  Menu,
  MenuDivider,
  MenuItem,
  MenuItemCheckbox,
  MenuList,
  MenuPopover,
  MenuTrigger,
  mergeClasses,
  tokens,
  useArrowNavigationGroup,
} from "@fluentui/react-components";
import { Checkmark12Filled } from "@fluentui/react-icons";
import {
  Fragment,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import * as ipc from "../common/ipc";
import { MenuItemData } from "../common/ipc";
import {
  formatAccelerator,
  getAccessKey,
  getVisibleItemIndices,
  parseLabel,
} from "./menuUtils";

/**
 * The menu bar drawn in the title bar on Windows and Linux,
 * which shows the application menu of the main process.
 */
export const MenuBar = (): ReactNode => {
  const styles = useStyles();
  const [menus, setMenus] = useState<MenuItemData[]>([]);
  const [openIndex, setOpenIndex] = useState<number>();
  const [showAccessKeys, setShowAccessKeys] = useState(false);
  const arrowNavigationGroup = useArrowNavigationGroup({
    axis: "horizontal",
    circular: true,
  });
  const barRef = useRef<HTMLDivElement>(null);
  const buttonRefs = useRef<HTMLElement[]>([]);
  const isActive = useRef(false);
  // Whether the menu bar was activated with the keyboard.
  const keyboardMode = useRef(false);
  // The element to focus when the menu bar is no longer used.
  const focusToReturn = useRef<HTMLElement | null>(null);

  const activate = useCallback(() => {
    if (isActive.current) return;
    isActive.current = true;
    focusToReturn.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
  }, []);

  const deactivate = useCallback((restoreFocus: boolean) => {
    isActive.current = false;
    keyboardMode.current = false;
    setOpenIndex(undefined);
    setShowAccessKeys(false);
    if (restoreFocus) {
      focusToReturn.current?.focus();
    }
    focusToReturn.current = null;
  }, []);

  const openMenu = useCallback(
    async (index: number) => {
      activate();
      // Get the latest states of the items, such as checkmarks.
      setMenus(
        await window.ipcRenderer.invoke<ipc.GetMainMenu>(ipc.getMainMenu),
      );
      setOpenIndex(index);
    },
    [activate],
  );

  const focusButton = useCallback(
    (index: number) => {
      activate();
      keyboardMode.current = true;
      setOpenIndex(undefined);
      setShowAccessKeys(true);
      buttonRefs.current[index]?.focus();
    },
    [activate],
  );

  const toggleFocus = useCallback(() => {
    if (isActive.current) {
      deactivate(true);
    } else {
      focusButton(0);
    }
  }, [deactivate, focusButton]);

  const findMenu = useCallback(
    (accessKey: string) =>
      menus.findIndex(
        (menu) => getAccessKey(menu.label) === accessKey.toLowerCase(),
      ),
    [menus],
  );

  const step = (index: number, delta: number) =>
    (index + delta + menus.length) % menus.length;

  const activateItem = (path: number[]) => {
    // Restore the focus first, so that, e.g., Edit > Paste applies to the focused input.
    deactivate(true);
    window.ipcRenderer.invoke<ipc.ClickMainMenuItem>(
      ipc.clickMainMenuItem,
      path,
    );
  };

  const renderList = (items: MenuItemData[], path: number[]) => {
    const indices = getVisibleItemIndices(items);
    return (
      <MenuList
        checkedValues={{
          checked: indices.filter((i) => items[i].checked).map(String),
        }}
        hasCheckmarks
        onKeyDownCapture={(e) => {
          // Also receives the events of the submenus, which are rendered in portals.
          if (!e.currentTarget.contains(e.target as Node)) return;
          if (e.altKey || e.ctrlKey || e.metaKey) return;
          const opensSubmenu = (e.target as Element).hasAttribute(
            "aria-haspopup",
          );
          if (
            path.length === 1 &&
            (e.key === "ArrowLeft" || (e.key === "ArrowRight" && !opensSubmenu))
          ) {
            e.preventDefault();
            setOpenIndex(step(path[0], e.key === "ArrowLeft" ? -1 : 1));
            return;
          }
          const item = e.currentTarget.querySelector<HTMLElement>(
            `[data-access-key="${CSS.escape(e.key.toLowerCase())}"]`,
          );
          if (!item) return;
          // Instead of moving the focus to the item that starts with the letter.
          e.preventDefault();
          e.stopPropagation();
          item.click();
        }}
      >
        {indices.map((i) => renderItem(items[i], [...path, i]))}
      </MenuList>
    );
  };

  const renderItem = (item: MenuItemData, path: number[]): ReactNode => {
    const key = path.at(-1);
    const props = {
      checkmark: { className: styles.itemCheckmark },
      className: styles.item,
      content: { className: styles.itemContent },
      "data-access-key": getAccessKey(item.label),
      disabled: !item.enabled,
      secondaryContent: item.accelerator && {
        children: formatAccelerator(item.accelerator),
        className: styles.itemSecondaryContent,
      },
    };
    const label = <Label label={item.label} showAccessKey={showAccessKeys} />;
    switch (item.type) {
      case "checkbox":
        return (
          <MenuItemCheckbox
            key={key}
            name="checked"
            value={String(key)}
            {...props}
            checkmark={{
              ...props.checkmark,
              // The default is for larger text.
              children: <Checkmark12Filled />,
            }}
            onClick={() => activateItem(path)}
          >
            {label}
          </MenuItemCheckbox>
        );
      case "separator":
        return <MenuDivider key={key} />;
      case "submenu":
        return (
          <Menu key={key}>
            <MenuTrigger disableButtonEnhancement>
              <MenuItem
                {...props}
                submenuIndicator={{ className: styles.itemSubmenuIndicator }}
              >
                {label}
              </MenuItem>
            </MenuTrigger>
            <MenuPopover>{renderList(item.submenu ?? [], path)}</MenuPopover>
          </Menu>
        );
      default:
        return (
          <MenuItem key={key} {...props} onClick={() => activateItem(path)}>
            {label}
          </MenuItem>
        );
    }
  };

  useEffect(() => {
    window.ipcRenderer.invoke<ipc.GetMainMenu>(ipc.getMainMenu).then(setMenus);
  }, []);

  useEffect(() => {
    // Whether Alt is pressed and released without any other key.
    let altAlone = false;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Alt") {
        altAlone = !e.repeat || altAlone;
        setShowAccessKeys(true);
        return;
      }
      altAlone = false;

      const noModifiers = !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey;
      if (e.key === "F10" && noModifiers) {
        e.preventDefault();
        toggleFocus();
        return;
      }

      const altOnly = e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey;
      // Alt+letter is also used for inserting symbols into relations.
      if (!altOnly || e.defaultPrevented) return;
      const index = findMenu(e.key);
      if (index === -1) return;
      e.preventDefault();
      keyboardMode.current = true;
      openMenu(index);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key !== "Alt") return;
      if (altAlone) {
        altAlone = false;
        e.preventDefault();
        toggleFocus();
      } else if (!isActive.current) {
        setShowAccessKeys(false);
      }
    };

    const onMouseDown = () => {
      altAlone = false;
    };

    const onBlur = () => {
      altAlone = false;
      if (isActive.current) {
        deactivate(false);
      } else {
        setShowAccessKeys(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("mousedown", onMouseDown, true);
      window.removeEventListener("blur", onBlur);
    };
  }, [deactivate, findMenu, openMenu, toggleFocus]);

  return (
    <div
      {...arrowNavigationGroup}
      className={styles.root}
      onBlur={(e) => {
        const to = e.relatedTarget;
        const staysInMenus =
          to instanceof Element &&
          (barRef.current?.contains(to) || to.closest("[role=menu]"));
        if (isActive.current && openIndex === undefined && !staysInMenus) {
          deactivate(false);
        }
      }}
      ref={barRef}
      role="menubar"
    >
      {menus.map((menu, index) => (
        <Fragment key={index}>
          <Button
            ref={(e) => {
              if (e) buttonRefs.current[index] = e;
            }}
            appearance="subtle"
            aria-expanded={openIndex === index}
            aria-haspopup="menu"
            className={mergeClasses(
              styles.button,
              openIndex === index && styles.openButton,
            )}
            onKeyDown={(e) => {
              const noModifiers =
                !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey;
              if (!noModifiers) return;
              switch (e.key) {
                case "ArrowDown":
                case "Enter":
                case " ":
                  openMenu(index);
                  break;
                case "Escape":
                  deactivate(true);
                  break;
                default: {
                  const i = findMenu(e.key);
                  if (i === -1) return;
                  openMenu(i);
                }
              }
              e.preventDefault();
            }}
            onMouseDown={(e) => {
              if (e.button !== 0) return;
              // Like a native menu bar, keep the focus, e.g., for Edit > Paste.
              e.preventDefault();
              if (openIndex === index) {
                deactivate(true);
              } else {
                openMenu(index);
              }
            }}
            onMouseEnter={() => {
              if (openIndex !== undefined) {
                setOpenIndex(index);
              }
            }}
            role="menuitem"
            size="small"
          >
            <Label label={menu.label} showAccessKey={showAccessKeys} />
          </Button>
          <Menu
            onOpenChange={(_, data) => {
              if (data.open) return;
              switch (data.type) {
                case "clickOutside":
                  // Clicks on the menu bar are handled by the buttons.
                  if (barRef.current?.contains(data.event.target as Node)) {
                    return;
                  }
                  // The click moves the focus.
                  deactivate(false);
                  return;
                case "menuItemClick":
                  return;
                case "menuPopoverKeyDown":
                  if (data.event.key === "Escape" && keyboardMode.current) {
                    focusButton(index);
                    return;
                  }
              }
              deactivate(true);
            }}
            open={openIndex === index}
            positioning={{
              align: "start",
              position: "below",
              target: {
                getBoundingClientRect: () =>
                  buttonRefs.current[index].getBoundingClientRect(),
              },
            }}
          >
            <MenuPopover>{renderList(menu.submenu ?? [], [index])}</MenuPopover>
          </Menu>
        </Fragment>
      ))}
    </div>
  );
};

const Label = ({
  label,
  showAccessKey,
}: {
  label: string;
  showAccessKey: boolean;
}): ReactNode => {
  const { accessKeyIndex: i, text } = parseLabel(label);
  if (!showAccessKey || i === undefined) {
    return text;
  }
  return (
    <>
      {text.slice(0, i)}
      <u>{text[i]}</u>
      {text.slice(i + 1)}
    </>
  );
};

const useStyles = makeStyles({
  root: {
    display: "flex",
  },
  button: {
    WebkitAppRegion: "no-drag",
    fontWeight: tokens.fontWeightRegular,
    minWidth: "auto",
  },
  openButton: {
    backgroundColor: tokens.colorSubtleBackgroundPressed,
  },
  // Same text size as the menu buttons. Like VS Code, the text is inset by 2em (24px) on both sides,
  // where the left inset holds the checkmark, and shortcuts are at least 4em away from the text.
  item: {
    fontSize: tokens.fontSizeBase200,
    lineHeight: tokens.lineHeightBase200,
    minHeight: "24px",
    paddingBottom: tokens.spacingVerticalXS,
    paddingLeft: "4px",
    paddingRight: "24px",
    paddingTop: tokens.spacingVerticalXS,
  },
  itemCheckmark: {
    alignItems: "center",
    display: "inline-flex",
    justifyContent: "center",
    marginTop: 0,
  },
  // The default is for larger text.
  itemSubmenuIndicator: {
    fontSize: "12px",
    height: "16px",
    width: "16px",
  },
  itemContent: {
    paddingLeft: 0,
    paddingRight: 0,
  },
  itemSecondaryContent: {
    lineHeight: tokens.lineHeightBase200,
    // 4em minus the gap between the slots of the item.
    paddingLeft: "44px",
    paddingRight: 0,
  },
});
