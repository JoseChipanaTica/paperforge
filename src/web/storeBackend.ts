/** The browser store's storage: three key-value stores. IndexedDB in the page, memory in tests. */
export type StoreName = "papers" | "artifacts" | "translations";

export interface Backend {
  get(store: StoreName, key: IDBValidKey): Promise<unknown>;
  put(store: StoreName, key: IDBValidKey, value: unknown): Promise<void>;
  delete(store: StoreName, key: IDBValidKey): Promise<void>;
  entries(store: StoreName): Promise<[IDBValidKey, unknown][]>;
  clear(): Promise<void>;
}

/** An in-memory backend; keys are compared by their JSON. */
export function memoryBackend(): Backend {
  const stores: Record<StoreName, Map<string, [IDBValidKey, unknown]>> = {
    papers: new Map(),
    artifacts: new Map(),
    translations: new Map(),
  };
  return {
    get: async (store, key) => stores[store].get(JSON.stringify(key))?.[1],
    put: async (store, key, value) => void stores[store].set(JSON.stringify(key), [key, value]),
    delete: async (store, key) => void stores[store].delete(JSON.stringify(key)),
    entries: async (store) => [...stores[store].values()],
    clear: async () => Object.values(stores).forEach((s) => s.clear()),
  };
}

/** No storage at all: every read misses, every write is dropped. */
export const nullBackend: Backend = {
  get: async () => undefined,
  put: async () => {},
  delete: async () => {},
  entries: async () => [],
  clear: async () => {},
};
