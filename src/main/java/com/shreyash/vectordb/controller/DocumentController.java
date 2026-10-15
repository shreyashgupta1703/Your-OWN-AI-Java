package com.shreyash.vectordb.controller;

import com.shreyash.vectordb.model.DocumentItem;
import com.shreyash.vectordb.service.*;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/doc")
@CrossOrigin(origins = "*")
public class DocumentController {
    private final DocumentDatabase db;
    private final OllamaClient ollama;
    private final TextChunker chunker;

    public DocumentController(DocumentDatabase db, OllamaClient ollama, TextChunker chunker) {
        this.db = db;
        this.ollama = ollama;
        this.chunker = chunker;
    }

    @PostMapping("/insert")
    public Object insert(@RequestBody InsertDocumentRequest request) {
        if (request.title() == null || request.title().isBlank()
                || request.text() == null || request.text().isBlank())
            return Map.of("error", "need title and text");

        List<String> chunks = chunker.chunk(request.text(), 250, 30);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] embedding = ollama.embed(chunks.get(i));
            if (embedding.length == 0)
                return Map.of("error", "Ollama unavailable. Run ollama serve and pull nomic-embed-text.");

            String title = chunks.size() > 1
                    ? request.title() + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : request.title();

            ids.add(db.insert(title, chunks.get(i), embedding));
        }

        return Map.of("ids", ids, "chunks", chunks.size(),
                "dims", chunks.isEmpty() ? 0 : ollama.embed("dimension check").length);
    }

    @DeleteMapping("/delete/{id}")
    public Map<String, Boolean> delete(@PathVariable int id) {
        return Map.of("ok", db.remove(id));
    }

    @GetMapping("/list")
    public List<Map<String, Object>> list() {
        return db.all().stream().map(d -> Map.<String, Object>of(
                "id", d.id(),
                "title", d.title(),
                "preview", d.text().length() > 120 ? d.text().substring(0, 120) + "…" : d.text(),
                "words", d.text().trim().isEmpty() ? 0 : d.text().trim().split("\\s+").length
        )).toList();
    }

    @PostMapping("/search")
    public Object search(@RequestBody AskRequest request) {
        float[] embedding = ollama.embed(request.question());
        if (embedding.length == 0) return Map.of("error", "Ollama unavailable");

        return Map.of("contexts", db.search(embedding, request.k()).stream()
                .map(r -> Map.of(
                        "id", r.document().id(),
                        "title", r.document().title(),
                        "distance", r.distance()))
                .toList());
    }

    @PostMapping("/ask")
    public Object ask(@RequestBody AskRequest request) {
        float[] embedding = ollama.embed(request.question());
        if (embedding.length == 0) return Map.of("error", "Ollama unavailable");

        List<DocumentDatabase.SearchResult> hits = db.search(embedding, request.k());

        StringBuilder context = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            context.append("[").append(i + 1).append("] ")
                    .append(hits.get(i).document().title()).append(":\n")
                    .append(hits.get(i).document().text()).append("\n\n");
        }

        String prompt = """
                You are a helpful assistant. Answer the user's question directly.
                Use the provided context if it contains relevant information.
                If it does not, use your general knowledge.
                Do not mention the context or say that the context does not mention something.

                Context:
                %s

                Question: %s

                Answer:
                """.formatted(context, request.question());

        String answer = ollama.generate(prompt);

        List<Map<String, Object>> contexts = hits.stream().map(r -> Map.<String, Object>of(
                "id", r.document().id(),
                "title", r.document().title(),
                "text", r.document().text(),
                "distance", r.distance()
        )).toList();

        return Map.of(
                "answer", answer,
                "model", ollama.genModel(),
                "contexts", contexts,
                "docCount", db.size()
        );
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        return Map.of(
                "ollamaAvailable", ollama.isAvailable(),
                "embedModel", ollama.embedModel(),
                "genModel", ollama.genModel(),
                "docCount", db.size()
        );
    }

    public record InsertDocumentRequest(String title, String text) {}
    public record AskRequest(String question, int k) {
        public AskRequest(String question) { this(question, 3); }
    }
}
