import { useEffect, useRef, useState } from "react";
import { queryCollectionStream } from "../api/client";
import type { SourceChunk } from "../api/client";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
}

interface Props {
  collection: string;
}

export default function ChatInterface({ collection }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(e: { preventDefault(): void }) {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    setLoading(true);

    // Helper to update the in-flight assistant message (always the last one).
    const updateAssistant = (patch: (m: Message) => Message) =>
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = patch(next[next.length - 1]);
        return next;
      });

    try {
      let started = false;
      await queryCollectionStream(collection, query, {
        onSources: (sources) => {
          setMessages((prev) => [...prev, { role: "assistant", content: "", sources }]);
          started = true;
        },
        onToken: (token) => {
          if (!started) {
            setMessages((prev) => [...prev, { role: "assistant", content: token }]);
            started = true;
          } else {
            updateAssistant((m) => ({ ...m, content: m.content + token }));
          }
        },
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${(err as Error).message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-gray-500 text-sm text-center mt-12">
            Ask a question about <span className="text-gray-400 font-medium">{collection}</span>.
          </p>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-gray-800 text-gray-100"
                }`}
              >
                {msg.content}
              </div>
            </div>

            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-2 space-y-1 ml-2">
                <p className="text-xs text-gray-500 mb-1">Sources</p>
                {msg.sources.map((s, j) => (
                  <details
                    key={j}
                    className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-xs text-gray-300"
                  >
                    <summary className="cursor-pointer text-gray-400 select-none">
                      {s.source}
                      {s.page_number != null ? ` · p.${s.page_number}` : ""}
                      {s.chunk_index != null ? ` · chunk ${s.chunk_index}` : ""}
                    </summary>
                    <p className="mt-2 text-gray-400 leading-relaxed">{s.content}</p>
                  </details>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && messages[messages.length - 1]?.role !== "assistant" && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-lg px-4 py-2 text-sm text-gray-400">
              Thinking…
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-800 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          disabled={loading}
          className="flex-1 bg-gray-800 text-sm text-gray-100 placeholder-gray-500 rounded-lg px-4 py-2 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm rounded-lg px-4 py-2 transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}
