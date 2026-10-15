package com.shreyash.vectordb.core;

import com.shreyash.vectordb.model.VectorItem;
import java.util.*;

public class KdTreeIndex {
    private static class Node {
        VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    public KdTreeIndex(int dims) { this.dims = dims; }

    public void insert(VectorItem item) {
        root = insert(root, item, 0);
    }

    private Node insert(Node node, VectorItem item, int depth) {
        if (node == null) return new Node(item);
        int axis = depth % dims;
        if (item.embedding()[axis] < node.item.embedding()[axis])
            node.left = insert(node.left, item, depth + 1);
        else
            node.right = insert(node.right, item, depth + 1);
        return node;
    }

    public List<Result> knn(float[] query, int k, Distance distance) {
        PriorityQueue<Result> heap = new PriorityQueue<>(
                Comparator.comparingDouble(Result::distance).reversed());
        search(root, query, k, 0, distance, heap);
        List<Result> out = new ArrayList<>(heap);
        out.sort(Comparator.comparingDouble(Result::distance));
        return out;
    }

    private void search(Node node, float[] q, int k, int depth,
                        Distance distance, PriorityQueue<Result> heap) {
        if (node == null) return;

        float d = distance.calculate(q, node.item.embedding());
        if (heap.size() < k || d < heap.peek().distance()) {
            heap.offer(new Result(d, node.item.id()));
            if (heap.size() > k) heap.poll();
        }

        int axis = depth % dims;
        float diff = q[axis] - node.item.embedding()[axis];

        Node near = diff < 0 ? node.left : node.right;
        Node far  = diff < 0 ? node.right : node.left;

        search(near, q, k, depth + 1, distance, heap);

        float worst = heap.isEmpty() ? Float.POSITIVE_INFINITY : heap.peek().distance();
        if (heap.size() < k || Math.abs(diff) < worst)
            search(far, q, k, depth + 1, distance, heap);
    }

    public void rebuild(Collection<VectorItem> items) {
        root = null;
        for (VectorItem item : items) insert(item);
    }

    public record Result(float distance, int id) {}
}
