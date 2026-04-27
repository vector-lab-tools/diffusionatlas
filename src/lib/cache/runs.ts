import { getDB } from "./idb";
import type { Run } from "@/types/run";

export async function saveRun(run: Run): Promise<void> {
  const db = await getDB();
  await db.put("runs", run, run.id);
}

export async function listRuns(): Promise<Run[]> {
  const db = await getDB();
  const tx = db.transaction("runs", "readonly");
  const store = tx.objectStore("runs");
  const all: Run[] = [];
  let cursor = await store.openCursor();
  while (cursor) {
    all.push(cursor.value as Run);
    cursor = await cursor.continue();
  }
  // Newest first.
  all.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return all;
}

export async function deleteRun(id: string): Promise<void> {
  const db = await getDB();
  await db.delete("runs", id);
}

export async function clearRuns(): Promise<void> {
  const db = await getDB();
  await db.clear("runs");
}
