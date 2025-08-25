# CS 370 — Treasure Hunt: Deep Q-Learning Pirate Agent

*A small maze. A patient pirate. A policy learned by trial, error, and grit.*

## Overview
This repository contains my Project Two artifact for CS 370: a Jupyter Notebook that trains a **Deep Q-Learning (DQN)** agent to guide a pirate NPC through a maze to the treasure. The agent learns via **ε-greedy exploration**, **experience replay**, and a **target network** for stability.

- Primary artifact: **`TreasureHuntGame.ipynb`**
- Language/stack: Python, NumPy, TensorFlow/Keras
- Starter files (provided by the course): `TreasureMaze.py` (environment) and `GameExperience.py` (experience replay utilities)

---

## What code was given vs. what I wrote
**Given (do not modify):**
- `TreasureMaze.py`: the maze environment (state observation, valid actions, act/step).
- `GameExperience.py`: replay buffer and helpers to build training targets.
- Notebook scaffolding: visualization, `build_model(maze)`, `play_game(...)`, and `completion_check(...)`.

**Created by me (in the notebook only):**
- The **Q-training loop** (`qtrain`): ε-greedy action selection, valid-action masking, storing transitions, sampling mini-batches from replay, fitting the network each step, and early-stopping when the rolling win-rate hits the threshold.
- Hyperparameters + seeds and readable progress logs every fixed number of epochs.
- Brief evaluation runs: completion check and a greedy playthrough.

---

## How to run
1. Open `TreasureHuntGame.ipynb` in Jupyter.
2. Run all cells (the training cell can take a while depending on hardware).
3. Verify:
   - `completion_check(model, qmaze)` returns `True`
   - `play_game(model, qmaze, pirate_cell=(0,0))` ends in **win**

---

## Results snapshot (optional numbers you can keep or update)
- Sliding win-rate repeatedly reached **1.000** in later epochs.
- By ~5,540 epochs, cumulative wins were **≈4,800 / 5,451 (~88%)** overall while the rolling window was often perfect.
- Loss typically hovered around **~0.0003–0.012**, with occasional spikes that did not derail performance.

---

## Reflection (Module Eight Journal)

### 1) Briefly explain the work you did
I integrated a **Deep Q-Learning** training loop into the given notebook—leveraging the environment (`TreasureMaze`) and replay utilities (`GameExperience`) without editing the `.py` files. My code implemented ε-greedy exploration, valid-action masking, replay sampling, model training with Huber loss + Adam, and an early-stop criterion based on a rolling win-rate. I then validated the learned policy with a greedy run and the provided `completion_check`.

### 2) Connect this course to the larger field
**What do computer scientists do and why does it matter?**  
We turn ambiguous problems into **well-specified systems**—designing data representations, choosing algorithms, and measuring outcomes. In AI, that means encoding goals, rewards, and constraints so learning systems can improve themselves. It matters because better decisions at scale—navigation, healthcare triage, accessibility tools—change lives.

**How do I approach a problem as a computer scientist?**  
1) **Frame** the task precisely (state, actions, rewards, termination).  
2) **Choose** a method aligned with constraints (here: DQN for discrete actions and compact states).  
3) **Prototype** the smallest thing that could work; instrument it with metrics.  
4) **Iterate**: tune hyperparameters, validate against edge cases, and document trade-offs.

**Ethical responsibilities to the end user and the organization**  
- **Transparency**: document assumptions, limits, and failure modes.  
- **Fairness & fun** (for games): NPCs should feel challenging but not exploit hidden bugs; no dark patterns.  
- **Safety & privacy**: avoid logging sensitive user data; minimize retention; respect platform policies.  
- **Sustainability**: use compute proportional to benefit; prefer efficient training loops and early stopping.  
- **Accountability**: version code and results; enable reproducibility.

---

## Repository contents
