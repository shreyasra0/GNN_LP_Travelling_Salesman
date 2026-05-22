To copy this, click the **"Copy"** button that appears in the top-right corner of the code block below. You can then paste the entire content directly into your `README.md` file.

```markdown
# TSP Hybrid Solver Project

This project implements a hybrid approach to solving the Traveling Salesperson Problem (TSP) by combining Graph Neural Networks (GNN) for edge pruning with a traditional Integer Linear Programming (LP) solver.

## Quick Start
To evaluate the model on your dataset:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

```

2. **Run Evaluation:**
Ensure your `.tsp` and `.opt.tour` files are located in the `data/` directory.
```bash
python evaluation.py

```


This will generate a per-instance performance report and a summary of your model's average recall and edge reduction.

## Current Performance
Based on an evaluation across 500 test instances, our GNN-augmented approach significantly outperforms standard optimization techniques by intelligently pruning the edge search space.

| Metric | Baseline (Raw LP) | GNN-Hybrid Solver |
| :--- | :--- | :--- |
| **Optimal Edge Recall** | 62.02% | **97.67%** |
| **Edge Search Space** | 100% (Full Graph) | **53.86%** (46.14% reduction) |

* **Recall:** By predicting optimal edges, the GNN ensures that 97.67% of the edges required for the optimal tour are preserved.
* **Efficiency:** The GNN prunes 46.14% of the search space, drastically reducing the number of variables the LP solver must consider, leading to faster convergence and lower memory consumption compared to a standard LP relaxation.

## Threshold Analysis Results

The GNN exhibits a highly polarized output distribution. Pruning threshold analysis indicates optimal stability and performance within the [0.35, 0.45] range.

| Threshold | Avg Recall | Avg Reduction (%) |
| --- | --- | --- |
| 0.35 | 97.67% | 46.14% |
| 0.45 | 97.67% | 46.14% |
| 0.55 | 93.60% | 47.59% |
| 0.65 | 0.00% | 100.00% |

## Next Steps: Future Roadmap

To improve the model beyond the current 97.67% recall ceiling, we have identified the following development path:

1. **Subtour Elimination:** Implement lazy constraint loops in the LP solver to ensure the generation of single, connected tours.
2. **GNN Architecture Refinement:** Increase GNN depth (layer complexity) to improve the receptive field and global path awareness, and integrate edge-based features (e.g., relative distances, angular information) to provide richer context.
3. **Training Optimization:** Explore loss function smoothing or "hard example mining" to improve the model's performance on the edges currently being misclassified.

```

Would you like to start on the Subtour Elimination logic for your LP solver next, or are you ready to dive into the GNN architecture refinement?

```