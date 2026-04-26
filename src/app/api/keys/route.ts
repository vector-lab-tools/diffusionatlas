/**
 * API Keys persistence — reads/writes API keys to .env.local
 * so they survive dev server restarts and browser storage clears.
 * .env.local is in .gitignore, so keys never reach GitHub.
 */

import { NextResponse } from "next/server";
import { readFile, writeFile } from "fs/promises";
import { join } from "path";

const ENV_PATH = join(process.cwd(), ".env.local");

interface KeysPayload {
  [key: string]: string;
}

const KEY_MAP: Record<string, string> = {
  replicate: "REPLICATE_API_TOKEN",
  fal: "FAL_KEY",
  together: "TOGETHER_API_KEY",
  stability: "STABILITY_API_KEY",
};

async function readEnvFile(): Promise<Map<string, string>> {
  const vars = new Map<string, string>();
  try {
    const content = await readFile(ENV_PATH, "utf-8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx < 0) continue;
      const key = trimmed.slice(0, eqIdx).trim();
      let val = trimmed.slice(eqIdx + 1).trim();
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      }
      vars.set(key, val);
    }
  } catch {
    /* file doesn't exist yet */
  }
  return vars;
}

async function writeEnvFile(vars: Map<string, string>): Promise<void> {
  const lines = ["# Diffusion Atlas — API Keys (auto-generated, do not commit)"];
  for (const [key, val] of vars) {
    lines.push(`${key}="${val}"`);
  }
  await writeFile(ENV_PATH, lines.join("\n") + "\n", "utf-8");
}

export async function GET() {
  const vars = await readEnvFile();
  const keys: KeysPayload = {};
  for (const [providerId, envVar] of Object.entries(KEY_MAP)) {
    const val = vars.get(envVar);
    if (val) keys[providerId] = val;
  }
  return NextResponse.json(keys);
}

export async function POST(req: Request) {
  const body: KeysPayload = await req.json();
  const vars = await readEnvFile();

  for (const [providerId, value] of Object.entries(body)) {
    const envVar = KEY_MAP[providerId];
    if (envVar && value) vars.set(envVar, value);
    else if (envVar && !value) vars.delete(envVar);
  }

  await writeEnvFile(vars);
  return NextResponse.json({ ok: true });
}
