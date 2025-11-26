import { notarize } from "@electron/notarize";
import "dotenv/config";
import * as path from "node:path";
import * as process from "node:process";

/** @type { (ctx: import("app-builder-lib").PackContext) => Promise<void> } */
export default async (ctx) => {
  if (ctx.electronPlatformName !== "darwin") return;

  const appPath = path.join(
    ctx.appOutDir,
    `${ctx.packager.appInfo.productFilename}.app`,
  );

  console.log(`Notarizing ${appPath}...`);

  const appleId = process.env.APPLE_ID;
  if (appleId === undefined) {
    throw new Error("Notarization failed: APPLE_ID must be set");
  }

  const appleIdPassword = process.env.APPLE_APP_SPECIFIC_PASSWORD;
  if (appleIdPassword === undefined) {
    throw new Error(
      "Notarization failed: APPLE_APP_SPECIFIC_PASSWORD must be set",
    );
  }

  const teamId = process.env.APPLE_TEAM_ID;
  if (teamId === undefined) {
    throw new Error("Notarization failed: APPLE_TEAM_ID must be set");
  }

  await notarize({ appPath, appleId, appleIdPassword, teamId });
};
