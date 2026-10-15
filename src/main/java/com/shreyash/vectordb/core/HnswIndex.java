package com.shreyash.vectordb.core;

import com.shreyash.vectordb.model.VectorItem;
import java.util.*;

public class HnswIndex {
    private static class Node {
        final VectorItem item;
        final int maxLayer;
        final List<List<Integer>> neighbors;

        Node(VectorItem item, int maxLayer) {
            this.item = item;
            this.maxLayer = maxLayer;
            this.neighbors = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++) neighbors.add(new ArrayList<>());
        }
    }

    public record Result(float distance, int id) {}
    public record GraphInfo(
            int topLayer,
            int nodeCount,
            List<Integer> nodesPerLayer,
            List<Integer> edgesPerLayer
    ) {}

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int m;
    private final int m0;
    private final int efConstruction;
    private final double levelMultiplier;
    private final Random random = new Random(42);

    private int entryPoint = -1;
    private int topLayer = -1;

    public HnswIndex() { this(16, 200); }

    public HnswIndex(int m, int efConstruction) {
        this.m = m;
        this.m0 = 2 * m;
        this.efConstruction = efConstruction;
        this.levelMultiplier = 1.0 / Math.log(m);
    }

    private int randomLevel() {
        double u = Math.max(random.nextDouble(), 1e-12);
        return (int) Math.floor(-Math.log(u) * levelMultiplier);
    }

    public synchronized void insert(VectorItem item, Distance distance) {
        int level = randomLevel();
        Node node = new Node(item, level);
        graph.put(item.id(), node);

        if (entryPoint == -1) {
            entryPoint = item.id();
            topLayer = level;
            return;
        }

        int ep = entryPoint;

        for (int layer = topLayer; layer > level; layer--) {
            if (layer < graph.get(ep).neighbors.size()) {
                List<Result> w = searchLayer(item.embedding(), ep, 1, layer, distance);
                if (!w.isEmpty()) ep = w.get(0).id();
            }
        }

        for (int layer = Math.min(topLayer, level); layer >= 0; layer--) {
            List<Result> w = searchLayer(item.embedding(), ep, efConstruction, layer, distance);
            int maxM = layer == 0 ? m0 : m;

            List<Integer> selected = w.stream()
                    .limit(maxM)
                    .map(Result::id)
                    .toList();

            node.neighbors.get(layer).addAll(selected);

            for (int neighborId : selected) {
                Node neighbor = graph.get(neighborId);
                if (neighbor == null) continue;

                while (neighbor.neighbors.size() <= layer)
                    neighbor.neighbors.add(new ArrayList<>());

                List<Integer> links = neighbor.neighbors.get(layer);
                if (!links.contains(item.id())) links.add(item.id());

                if (links.size() > maxM) {
                    links.sort(Comparator.comparingDouble(
                            id -> distance.calculate(neighbor.item.embedding(), graph.get(id).item.embedding())));
                    while (links.size() > maxM) links.remove(links.size() - 1);
                }
            }

            if (!w.isEmpty()) ep = w.get(0).id();
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = item.id();
        }
    }

    private List<Result> searchLayer(float[] query, int entry, int ef,
                                     int layer, Distance distance) {
        Node start = graph.get(entry);
        if (start == null || layer >= start.neighbors.size()) return List.of();

        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Result> candidates =
                new PriorityQueue<>(Comparator.comparingDouble(Result::distance));
        PriorityQueue<Result> found =
                new PriorityQueue<>(Comparator.comparingDouble(Result::distance).reversed());

        float d0 = distance.calculate(query, start.item.embedding());
        visited.add(entry);
        candidates.offer(new Result(d0, entry));
        found.offer(new Result(d0, entry));

        while (!candidates.isEmpty()) {
            Result current = candidates.poll();

            if (found.size() >= ef && current.distance() > found.peek().distance())
                break;

            Node currentNode = graph.get(current.id());
            if (currentNode == null || layer >= currentNode.neighbors.size()) continue;

            for (int neighborId : currentNode.neighbors.get(layer)) {
                if (!visited.add(neighborId)) continue;
                Node neighbor = graph.get(neighborId);
                if (neighbor == null) continue;

                float d = distance.calculate(query, neighbor.item.embedding());

                if (found.size() < ef || d < found.peek().distance()) {
                    candidates.offer(new Result(d, neighborId));
                    found.offer(new Result(d, neighborId));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<Result> result = new ArrayList<>(found);
        result.sort(Comparator.comparingDouble(Result::distance));
        return result;
    }

    public synchronized List<Result> knn(float[] query, int k, int ef, Distance distance) {
        if (entryPoint == -1) return List.of();

        int ep = entryPoint;
        for (int layer = topLayer; layer > 0; layer--) {
            List<Result> w = searchLayer(query, ep, 1, layer, distance);
            if (!w.isEmpty()) ep = w.get(0).id();
        }

        List<Result> result = searchLayer(query, ep, Math.max(ef, k), 0, distance);
        return result.subList(0, Math.min(k, result.size()));
    }

    public synchronized void remove(int id) {
        if (!graph.containsKey(id)) return;

        for (Node node : graph.values())
            for (List<Integer> links : node.neighbors)
                links.removeIf(x -> x == id);

        graph.remove(id);

        if (id == entryPoint) {
            if (graph.isEmpty()) {
                entryPoint = -1;
                topLayer = -1;
            } else {
                Node first = graph.values().iterator().next();
                entryPoint = first.item.id();
                topLayer = first.maxLayer;
            }
        }
    }

    public synchronized GraphInfo graphInfo() {
        int layers = Math.max(topLayer + 1, 1);
        List<Integer> nodes = new ArrayList<>(Collections.nCopies(layers, 0));
        List<Integer> edges = new ArrayList<>(Collections.nCopies(layers, 0));

        for (Node node : graph.values()) {
            for (int layer = 0; layer <= node.maxLayer && layer < layers; layer++) {
                nodes.set(layer, nodes.get(layer) + 1);
                if (layer < node.neighbors.size())
                    edges.set(layer, edges.get(layer) + node.neighbors.get(layer).size());
            }
        }
        return new GraphInfo(topLayer, graph.size(), nodes, edges);
    }

    public int size() { return graph.size(); }
}
