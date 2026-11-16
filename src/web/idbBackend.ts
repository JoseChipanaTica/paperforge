import { type Backend, type StoreName, nullBackend } from "./storeBackend";

const DB_NAME = "paperforge";
const DB_VERSION = 1;
const STORES: StoreName[] = ["papers", "artifacts", "translations"];

/** A transaction's end: commit failures such as QuotaExceededError arrive as an abort after the request succeeded. */
const committed = (tx: IDBTransaction) =>
  new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new DOMException("Transaction aborted", "AbortError"));
  });

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
      let blocked = false;
      req.onsuccess = () => (blocked ? req.result.close() : resolve(req.result));
      req.onerror = () => reject(req.error);
      req.onblocked = () => {
        blocked = true;
        reject(new Error("IndexedDB open blocked"));
      };
    });
  } catch {
    return nullBackend;
  }
  // Another tab upgrading the schema: let it, this tab stops saving (reads/writes then reject → misses/no-ops).
  db.onversionchange = () => db.close();

  const store = (name: StoreName, mode: IDBTransactionMode) => db.transaction(name, mode).objectStore(name);
  /** A write counts once its transaction commits; the request's own success is not enough. */
  const write = (name: StoreName, op: (s: IDBObjectStore) => IDBRequest) => {
    const tx = db.transaction(name, "readwrite");
    const end = committed(tx);
    op(tx.objectStore(name));
    return end;
  };
  return {
    get: (name, key) => done(store(name, "readonly").get(key)),
    put: (name, key, value) => write(name, (s) => s.put(value, key)),
    delete: (name, key) => write(name, (s) => s.delete(key)),
    entries: async (name) => {
      const s = store(name, "readonly");
      const [keys, values] = await Promise.all([done(s.getAllKeys()), done(s.getAll())]);
      return keys.map((k, i) => [k, values[i]] as [IDBValidKey, unknown]);
    },
    clear: async () => {
      const tx = db.transaction(STORES, "readwrite");
      const end = committed(tx);
      for (const name of STORES) tx.objectStore(name).clear();
      await end;
    },
  };
}
