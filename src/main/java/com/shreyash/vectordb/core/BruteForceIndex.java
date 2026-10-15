package com.shreyash.vectordb.core;

import com.shreyash.vectordb.model.VectorItem;
import java.util.*;

public class BruteForceIndex {
    private final Map<Integer, VectorItem> items = new LinkedHashMap<>();

    public void insert(VectorItem item) { items.put(item.id(), item); }
    public void remove(int id) { items.remove(id); }

    public List<Result> knn(float[] query, int k, Distance distance) {
        return items.values().stream()
                .map(v -> new Result(distance.calculate(query, v.embedding()), v.id()))
                .sorted(Comparator.comparingDouble(Result::distance))
                .limit(k)
                .toList();
    }

    public record Result(float distance, int id) {}
    public Collection<VectorItem> values() { return items.values(); }
}
