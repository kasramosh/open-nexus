import { useState } from "react";
import type { Collection } from "../api/client";

interface Props {
  collections: Collection[];
  selected: string | null;
  onSelect: (name: string) => void;
  onCreate: (name: string) => Promise<void>;
  onDelete: (name: string) => Promise<void>;
}

export default function CollectionList({
  collections,
  selected,
  onSelect,
  onCreate,
  onDelete,
}: Props) {
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCreate(e: { preventDefault(): void }) {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      await onCreate(newName.trim());
      setNewName("");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="w-64 flex flex-col bg-gray-900 border-r border-gray-800 h-full shrink-0">
      <div className="p-4 border-b border-gray-800">
        <h1 className="text-lg font-semibold text-white">Nexus</h1>
        <p className="text-xs text-gray-500 mt-0.5">Collections</p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {collections.length === 0 ? (
          <p className="text-xs text-gray-500 p-2">No collections yet.</p>
        ) : (
          collections.map((c) => (
            <div
              key={c.name}
              onClick={() => onSelect(c.name)}
              className={`flex items-center justify-between rounded px-3 py-2 cursor-pointer group mb-1 ${
                selected === c.name
                  ? "bg-indigo-600 text-white"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
            >
              <span className="text-sm truncate">{c.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(c.name);
                }}
                title="Delete collection"
                className="text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity text-xs ml-2 shrink-0"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>

      <form onSubmit={handleCreate} className="p-3 border-t border-gray-800 space-y-2">
        {error && <p className="text-xs text-red-400">{error}</p>}
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New collection name"
          className="w-full bg-gray-800 text-sm text-gray-100 placeholder-gray-500 rounded px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
        <p className="text-xs text-gray-600">
          3–80 chars · letters, numbers, - and _ · must start and end with a letter or number
        </p>
        <button
          type="submit"
          disabled={creating || !newName.trim()}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm rounded px-3 py-1.5 transition-colors"
        >
          {creating ? "Creating…" : "Create"}
        </button>
      </form>
    </div>
  );
}
