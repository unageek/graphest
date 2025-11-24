import { Buffer } from "node:buffer";

export function fromBase64Url(base64Url: string): string {
  const buf = Buffer.from(base64Url, "base64url");
  return buf.toString("utf8");
}

export function toBase64Url(string: string): string {
  const buf = Buffer.from(string, "utf8");
  return buf.toString("base64url");
}
