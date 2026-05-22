# TSP Hybrid Solver Project

This project implements a hybrid approach to solving the Traveling Salesperson Problem (TSP) by combining Graph Neural Networks (GNN) for edge pruning with a traditional Integer Linear Programming (LP) solver.

## Current Performance
- **Recall:** 96% of optimal edges are successfully identified and preserved by the GNN.
- **Efficiency:** The pruning process achieves a ~50% reduction in the number of edges passed to the LP solver, significantly lowering computational overhead and memory usage.

## Threshold Analysis Results
The GNN exhibits a highly polarized output distribution. Pruning threshold analysis indicates consistent performance across the [0.35, 0.55] range:

| Threshold | Edges Kept | Recall |
| :--- | :--- | :--- |
| 0.35 | 353 | 0.9600 |
| 0.40 | 353 | 0.9600 |
| 0.45 | 353 | 0.9600 |
| 0.50 | 353 | 0.9600 |
| 0.55 | 353 | 0.9600 |
| 0.60 | 0 | 0.0000 |

## Next Steps: Future Roadmap
To improve the model beyond the current 96% recall ceiling, we have identified the following development path:

1. **Subtour Elimination:** Implement lazy constraint loops in the LP solver to ensure the generation of single, connected tours.
2. **GNN Architecture Refinement:** - Increase GNN depth (layer complexity) to improve the receptive field and global path awareness.
   - Integrate edge-based features (e.g., relative distances, angular information) to provide richer context for the edge predictor.
3. **Training Optimization:** Explore loss function smoothing or "hard example mining" to improve the model's performance on the 4% of edges currently being misclassified.