import { type Backend, type StoreName, nullBackend } from "./storeBackend";

const DB_NAME = "paperforge";
const DB_VERSION = 1;
const STORES: StoreName[] = ["papers", "artifacts", "translations"];

const done = <T>(req: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

/** The IndexedDB backend, or `nullBackend` when IndexedDB is missing, refuses to open or is blocked. */
export async function openIdbBackend(): Promise<Backend> {
  if (typeof indexedDB === "undefined") return nullBackend;
  let db: IDBDatabase;
  try {
    db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        for (const name of STORES) if (!req.result.objectStoreNames.contains(name)) req.result.createObjectStore(name);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error("IndexedDB open blocked"));
    });
  } catch {
    return nullBackend;
  }
  // Another tab upgrading the schema: let it, this tab stops saving (reads/writes then reject → misses/no-ops).
  db.onversionchange = () => db.close();

  const store = (name: StoreName, mode: IDBTransactionMode) => db.transaction(name, mode).objectStore(name);
  return {
    get: (name, key) => done(store(name, "readonly").get(key)),
    put: async (name, key, value) => void (await done(store(name, "readwrite").put(value, key))),
    delete: async (name, key) => void (await done(store(name, "readwrite").delete(key))),
    entries: async (name) => {
      const s = store(name, "readonly");
      const [keys, values] = await Promise.all([done(s.getAllKeys()), done(s.getAll())]);
      return keys.map((k, i) => [k, values[i]] as [IDBValidKey, unknown]);
    },
    clear: async () => {
      const tx = db.transaction(STORES, "readwrite");
      for (const name of STORES) tx.objectStore(name).clear();
      await new Promise<void>((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      });
    },
  };
}
