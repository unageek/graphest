import { notarize } from "@electron/notarize";
import "dotenv/config";
import * as path from "node:path";
import * as process from "node:process";

/** @type { (ctx: import("app-builder-lib").PackContext) => Promise<void> } */
export default async function (ctx) {
  if (ctx.electronPlatformName !== "darwin") return;

  const appPath = path.join(
    ctx.appOutDir,
    `${ctx.packager.appInfo.productFilename}.app`,
  );

  console.log(`Notarizing ${appPath}...`);

  await notarize({
    appPath,
    appleId: /** @type { string } */ (process.env.notarizeAppleId),
    appleIdPassword: /** @type { string } */ (
      process.env.notarizeAppleIdPassword
    ),
    teamId: /** @type { string } */ (process.env.notarizeTeamId),
  });
}
