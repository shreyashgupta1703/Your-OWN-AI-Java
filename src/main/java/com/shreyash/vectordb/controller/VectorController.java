package com.shreyash.vectordb.controller;

import com.shreyash.vectordb.model.VectorItem;
import com.shreyash.vectordb.service.VectorDatabase;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping
@CrossOrigin(origins = "*")
public class VectorController {
    private final VectorDatabase db;

    public VectorController(VectorDatabase db) { this.db = db; }

    @GetMapping("/search")
    public VectorDatabase.SearchResponse search(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric,
            @RequestParam(defaultValue = "hnsw") String algo) {
        return db.search(parseVector(v), k, metric, algo);
    }

    @PostMapping("/insert")
    public Map<String, Integer> insert(@RequestBody InsertRequest request) {
        return Map.of("id", db.insert(request.metadata(), request.category(), request.embedding()));
    }

    @DeleteMapping("/delete/{id}")
    public Map<String, Boolean> delete(@PathVariable int id) {
        return Map.of("ok", db.remove(id));
    }

    @GetMapping("/items")
    public Collection<VectorItem> items() { return db.all(); }

    @GetMapping("/benchmark")
    public VectorDatabase.BenchmarkResponse benchmark(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric) {
        return db.benchmark(parseVector(v), k, metric);
    }

    @GetMapping("/hnsw-info")
    public Object hnswInfo() { return db.hnswInfo(); }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return Map.of(
                "count", db.size(),
                "dims", 16,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics", List.of("euclidean", "cosine", "manhattan")
        );
    }

    private float[] parseVector(String value) {
        String[] parts = value.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Float.parseFloat(parts[i].trim());
        return out;
    }

    public record InsertRequest(String metadata, String category, float[] embedding) {}
}
