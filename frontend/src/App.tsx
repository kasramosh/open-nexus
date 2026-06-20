import { useEffect, useState } from "react";
import {
  createCollection,
  deleteCollection,
  getCollections,
  getDocuments,
} from "./api/client";
import type { Collection, DocumentItem } from "./api/client";
import CollectionList from "./components/CollectionList";
import DocumentUpload from "./components/DocumentUpload";
import ChatInterface from "./components/ChatInterface";

export default function App() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // Load collections once on mount
  useEffect(() => {
    getCollections().then(setCollections).catch(console.error);
  }, []);

  // Load documents whenever the selected collection changes. The cleanup flag
  // guards against a stale response from a previous collection landing late.
  useEffect(() => {
    if (!selected) return;
    let active = true;
    getDocuments(selected)
      .then((docs) => {
        if (active) setDocuments(docs);
      })
      .catch(console.error);
    return () => {
      active = false;
    };
  }, [selected]);

  async function handleCreate(name: string) {
    await createCollection(name);
    const updated = await getCollections();
    setCollections(updated);
    setSelected(name);
  }

  async function handleDelete(name: string) {
    await deleteCollection(name);
    const updated = await getCollections();
    setCollections(updated);
    if (selected === name) {
      const next = updated[0]?.name ?? null;
      setSelected(next);
      if (!next) setDocuments([]);
    }
  }

  function refreshDocuments() {
    if (selected) getDocuments(selected).then(setDocuments).catch(console.error);
  }

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <CollectionList
        collections={collections}
        selected={selected}
        onSelect={setSelected}
        onCreate={handleCreate}
        onDelete={handleDelete}
      />

      {selected ? (
        <div className="flex flex-col flex-1 overflow-hidden">
          <DocumentUpload
            collection={selected}
            documents={documents}
            onRefresh={refreshDocuments}
          />
          {/* key={selected} makes React fully remount ChatInterface when collection
              changes, which clears the message history automatically */}
          <ChatInterface key={selected} collection={selected} />
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
          Select or create a collection to get started.
        </div>
      )}
    </div>
  );
}
