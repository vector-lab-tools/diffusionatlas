import { openDB, type IDBPDatabase } from "idb";

const DB_NAME = "diffusion-atlas";
const DB_VERSION = 2;

let dbPromise: Promise<IDBPDatabase> | null = null;

export function getDB(): Promise<IDBPDatabase> {
  if (!dbPromise) {
    dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains("latents")) db.createObjectStore("latents");
        if (!db.objectStoreNames.contains("images")) db.createObjectStore("images");
        if (!db.objectStoreNames.contains("runs")) db.createObjectStore("runs");
        if (!db.objectStoreNames.contains("bench")) db.createObjectStore("bench");
      },
    });
  }
  return dbPromise;
}
