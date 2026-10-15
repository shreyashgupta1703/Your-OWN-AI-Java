package com.shreyash.vectordb.service;

import com.shreyash.vectordb.core.*;
import com.shreyash.vectordb.model.VectorItem;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class VectorDatabase {
    private static final int DIMS = 16;

    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForceIndex brute = new BruteForceIndex();
    private final KdTreeIndex kdTree = new KdTreeIndex(DIMS);
    private final HnswIndex hnsw = new HnswIndex(16, 200);
    private final AtomicInteger nextId = new AtomicInteger(1);

    public VectorDatabase() {
        loadDemo();
    }

    public synchronized int insert(String metadata, String category, float[] embedding) {
        if (embedding.length != DIMS)
            throw new IllegalArgumentException("Expected " + DIMS + " dimensions");

        VectorItem item = new VectorItem(nextId.getAndIncrement(), metadata, category, embedding);
        store.put(item.id(), item);
        brute.insert(item);
        kdTree.insert(item);
        hnsw.insert(item, DistanceMetrics::cosine);
        return item.id();
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        brute.remove(id);
        hnsw.remove(id);
        kdTree.rebuild(store.values());
        return true;
    }

    public synchronized SearchResponse search(float[] query, int k, String metric, String algorithm) {
        Distance distance = DistanceMetrics.of(metric);
        long start = System.nanoTime();

        List<? extends Object> raw = switch (algorithm.toLowerCase()) {
            case "bruteforce", "brute" -> brute.knn(query, k, distance);
            case "kdtree", "kd-tree" -> kdTree.knn(query, k, distance);
            default -> hnsw.knn(query, k, 50, distance);
        };

        List<SearchHit> hits = new ArrayList<>();
        for (Object obj : raw) {
            int id;
            float d;
            if (obj instanceof BruteForceIndex.Result r) { id = r.id(); d = r.distance(); }
            else if (obj instanceof KdTreeIndex.Result r) { id = r.id(); d = r.distance(); }
            else { HnswIndex.Result r = (HnswIndex.Result)obj; id = r.id(); d = r.distance(); }

            VectorItem item = store.get(id);
            if (item != null)
                hits.add(new SearchHit(id, item.metadata(), item.category(), item.embedding(), d));
        }

        long micros = (System.nanoTime() - start) / 1_000;
        return new SearchResponse(hits, micros, algorithm, metric);
    }

    public synchronized BenchmarkResponse benchmark(float[] query, int k, String metric) {
        Distance distance = DistanceMetrics.of(metric);

        long t = System.nanoTime(); brute.knn(query, k, distance);
        long bf = (System.nanoTime() - t) / 1_000;

        t = System.nanoTime(); kdTree.knn(query, k, distance);
        long kd = (System.nanoTime() - t) / 1_000;

        t = System.nanoTime(); hnsw.knn(query, k, 50, distance);
        long hn = (System.nanoTime() - t) / 1_000;

        return new BenchmarkResponse(bf, kd, hn, store.size());
    }

    public synchronized Collection<VectorItem> all() { return new ArrayList<>(store.values()); }
    public synchronized int size() { return store.size(); }
    public HnswIndex.GraphInfo hnswInfo() { return hnsw.graphInfo(); }

    public record SearchHit(int id, String metadata, String category, float[] embedding, float distance) {}
    public record SearchResponse(List<SearchHit> results, long latencyUs, String algo, String metric) {}
    public record BenchmarkResponse(long bruteforceUs, long kdtreeUs, long hnswUs, int itemCount) {}

    private void loadDemo() {
        insert("Linked List: nodes connected by pointers", "cs",
                new float[]{.90f,.85f,.72f,.68f,.12f,.08f,.15f,.10f,.05f,.08f,.06f,.09f,.07f,.11f,.08f,.06f});
        insert("Binary Search Tree: O(log n) search and insert", "cs",
                new float[]{.88f,.82f,.78f,.74f,.15f,.10f,.08f,.12f,.06f,.07f,.08f,.05f,.09f,.06f,.07f,.10f});
        insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                new float[]{.82f,.76f,.88f,.80f,.20f,.18f,.12f,.09f,.07f,.06f,.08f,.07f,.08f,.09f,.06f,.07f});
        insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                new float[]{.85f,.80f,.75f,.82f,.18f,.14f,.10f,.08f,.06f,.09f,.07f,.06f,.10f,.08f,.09f,.07f});
        insert("Hash Table: O(1) lookup with collision chaining", "cs",
                new float[]{.87f,.78f,.70f,.76f,.13f,.11f,.09f,.14f,.08f,.07f,.06f,.08f,.07f,.10f,.08f,.09f});

        insert("Calculus: derivatives integrals and limits", "math",
                new float[]{.12f,.15f,.18f,.10f,.91f,.86f,.78f,.72f,.08f,.06f,.07f,.09f,.07f,.08f,.06f,.10f});
        insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                new float[]{.20f,.18f,.15f,.12f,.88f,.90f,.82f,.76f,.09f,.07f,.08f,.06f,.10f,.07f,.08f,.09f});
        insert("Probability: distributions random variables Bayes theorem", "math",
                new float[]{.15f,.12f,.20f,.18f,.84f,.80f,.88f,.82f,.07f,.08f,.06f,.10f,.09f,.06f,.09f,.08f});
        insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                new float[]{.22f,.16f,.14f,.20f,.80f,.85f,.76f,.90f,.08f,.09f,.07f,.06f,.08f,.10f,.07f,.06f});
        insert("Combinatorics: permutations combinations generating functions", "math",
                new float[]{.18f,.20f,.16f,.14f,.86f,.78f,.84f,.80f,.06f,.07f,.09f,.08f,.06f,.09f,.10f,.07f});

        insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                new float[]{.08f,.06f,.09f,.07f,.07f,.08f,.06f,.09f,.90f,.86f,.78f,.72f,.08f,.06f,.09f,.07f});
        insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                new float[]{.06f,.08f,.07f,.09f,.09f,.06f,.08f,.07f,.86f,.90f,.82f,.76f,.07f,.09f,.06f,.08f});
        insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                new float[]{.09f,.07f,.06f,.08f,.08f,.09f,.07f,.06f,.82f,.78f,.90f,.84f,.09f,.07f,.08f,.06f});
        insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                new float[]{.07f,.09f,.08f,.06f,.06f,.07f,.09f,.08f,.78f,.82f,.86f,.90f,.06f,.08f,.07f,.09f});
        insert("Croissant: laminated pastry with buttery flaky layers", "food",
                new float[]{.06f,.07f,.10f,.09f,.10f,.06f,.07f,.10f,.85f,.80f,.76f,.82f,.09f,.07f,.10f,.06f});

        insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                new float[]{.09f,.07f,.08f,.10f,.08f,.09f,.07f,.06f,.08f,.07f,.09f,.06f,.91f,.85f,.78f,.72f});
        insert("Football: tackles touchdowns field goals and strategy", "sports",
                new float[]{.07f,.09f,.06f,.08f,.09f,.07f,.10f,.08f,.07f,.09f,.08f,.07f,.87f,.89f,.82f,.76f});
        insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                new float[]{.08f,.06f,.09f,.07f,.07f,.08f,.06f,.09f,.09f,.06f,.07f,.08f,.83f,.80f,.88f,.82f});
        insert("Chess: openings endgames tactics strategic board game", "sports",
                new float[]{.25f,.20f,.22f,.18f,.22f,.18f,.20f,.15f,.06f,.08f,.07f,.09f,.80f,.84f,.78f,.90f});
        insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                new float[]{.06f,.08f,.07f,.09f,.08f,.06f,.09f,.07f,.10f,.08f,.06f,.07f,.85f,.82f,.86f,.80f});
    }
}
