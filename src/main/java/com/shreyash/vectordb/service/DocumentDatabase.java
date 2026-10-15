package com.shreyash.vectordb.service;

import com.shreyash.vectordb.core.*;
import com.shreyash.vectordb.model.*;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class DocumentDatabase {
    private final Map<Integer, DocumentItem> store = new LinkedHashMap<>();
    private final HnswIndex hnsw = new HnswIndex(16, 200);
    private final BruteForceIndex brute = new BruteForceIndex();
    private final AtomicInteger nextId = new AtomicInteger(1);

    public synchronized int insert(String title, String text, float[] embedding) {
        DocumentItem item = new DocumentItem(nextId.getAndIncrement(), title, text, embedding);
        store.put(item.id(), item);
        VectorItem vector = new VectorItem(item.id(), title, "doc", embedding);
        hnsw.insert(vector, DistanceMetrics::cosine);
        brute.insert(vector);
        return item.id();
    }

    public synchronized List<SearchResult> search(float[] query, int k) {
        List<HnswIndex.Result> raw = store.size() < 10
                ? brute.knn(query, k, DistanceMetrics::cosine).stream()
                    .map(r -> new HnswIndex.Result(r.distance(), r.id())).toList()
                : hnsw.knn(query, k, 50, DistanceMetrics::cosine);

        List<SearchResult> result = new ArrayList<>();
        for (HnswIndex.Result r : raw) {
            DocumentItem item = store.get(r.id());
            if (item != null && r.distance() <= 0.7f)
                result.add(new SearchResult(r.distance(), item));
        }
        return result;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        hnsw.remove(id);
        brute.remove(id);
        return true;
    }

    public synchronized Collection<DocumentItem> all() { return new ArrayList<>(store.values()); }
    public synchronized int size() { return store.size(); }

    public record SearchResult(float distance, DocumentItem document) {}
}
