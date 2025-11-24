import { notarize } from "@electron/notarize";
import "dotenv/config";
import * as path from "node:path";
import * as process from "node:process";

export default async function (ctx) {
  if (ctx.electronPlatformName !== "darwin") return;

  const appPath = path.join(
    ctx.appOutDir,
    `${ctx.packager.appInfo.productFilename}.app`
  );

  console.log(`Notarizing ${appPath}...`);

  await notarize({
    appPath,
    appleId: process.env.notarizeAppleId,
    appleIdPassword: process.env.notarizeAppleIdPassword,
    teamId: process.env.notarizeTeamId,
  });
}
