# SBL-Inspired Bidirectional Planner with Lazy Collision Checking

## Overview

This repository contains the implementation and evaluation of a Single-query, Bidirectional, Lazy Collision Checking (SBL) planner for robotic motion planning. Unlike multi-query approaches, this algorithm targets a single start-to-goal problem to minimize computational effort.

The project was developed by Christian Mentenich, Gabrielle Mentenich and Michael Poncelet as part of the KIT lecture **"Innovative Konzepte zur Programmierung von Industrierobotern"** (Prof. Dr.-Ing. Björn Hein). It combines bidirectional tree growth, lazy evaluation, and adaptive local collision checking to efficiently find valid paths for both 2-DoF point robots and n-DoF planar manipulators.

---

## Repository Structure

The codebase is organized as follows to separate the core logic, helper functions, tests, and the final technical report:

* **`notebooks/SBL.ipynb`**: The main Jupyter Notebook serving as the technical report. It contains the step-by-step explanation, visualizations, benchmarking comparisons (against RRT and LazyPRM), and statistical evaluations of the planner.


* **`planners/SBL.py`**: The core Python implementation of the bidirectional SBL planner.


* **`modules/`** and **`lecture_examples/`**: Directories containing supporting code, such as environment definitions, kinematics (`IPEnvironmentKin.py`), planar manipulator models, and performance monitoring tools.


* **`tests/`**: Unit tests validating crucial components like the adaptive line test, graph structures, and collision-free path verification.



---

## Key Features

* **Bidirectional Tree Growth:** The planner simultaneously grows two search trees—one from the start configuration and one from the goal configuration.


* **Lazy Collision Checking:** Edge collision checks are delayed and stored as `unknown` until a potential connection between the two trees is identified. Only edges needed for a candidate path are validated, and previously validated edges are reused.


* **Adaptive Local Collision Validation:** A recursive midpoint checking strategy is used for edge validation. It tests the midpoint first, recursively subdivides intervals down to a resolution of `epsilon`, and aborts immediately upon detecting a collision.


* **N-DoF Support:** Designed to handle complex, multi-dimensional configuration spaces beyond standard 2-DoF environments.


* **Comprehensive Visualization:** Includes visual representations of the start and goal trees, unchecked/valid/invalid edges, candidate paths, and the exact points tested during adaptive collision checking.



---

## Getting Started

### 1. Running the Planner and Experiments

The primary entry point for exploring the planner, running experiments, and viewing results is the main notebook. Launch Jupyter and open the file:

```bash
jupyter notebook notebooks/SBL.ipynb

```

### 2. Running the Tests

To ensure the integrity of the planner and the adaptive collision checking logic, run the unit tests located in the `tests/` directory:

```bash
pytest tests/

```

---

## Evaluation & Benchmarks

The planner has been rigorously evaluated across various scenarios:

1. **Environments:** Tested in 2-DoF environments (open spaces, narrow passages, trap-like environments) and n-DoF configurations (2-DoF and 4-DoF planar manipulators).


2. **Metrics:** Benchmarks include success rates, planning times, node generation counts, and the exact number of point vs. line tests performed.


3. **Comparisons:** The SBL planner's performance is compared against baseline algorithms such as RRT and LazyPRM.



---

## Team & Division of Labor

This project was completed by a group of three students. For specific details regarding the division of labor (e.g., tree growth logic, lazy validation, and statistical evaluation), please refer to the corresponding section in `notebooks/SBL.ipynb`.
