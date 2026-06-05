"""
22.3-bandits.py — Implémentation complète et exécutable des algorithmes de bandits multi-bras.

Reproduit le "10-armed testbed" de Sutton & Barto (Chapitre 2) et compare 6 agents :
  - ε-greedy (ε=0.1)
  - ε-greedy (ε=0.01)
  - Optimistic Initial Values (Q₀=5)
  - UCB (c=2)
  - Thompson Sampling (gaussien)
  - Gradient Bandit (α=0.1)

Usage:
    python 22.3-bandits.py

Génère le fichier `bandits_comparison.png` avec les courbes de récompense moyenne
et de pourcentage d'actions optimales pour chaque agent.

Auteur : Dr. Haythem REHOUMA — Cours d'Apprentissage par Renforcement
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. ENVIRONNEMENT
# ============================================================================


class BanditEnvironment:
    """
    Environnement de bandit multi-bras stationnaire.

    À l'initialisation, la vraie valeur q*(a) de chaque bras est tirée
    d'une loi normale N(0, 1). À chaque appel à step(action), la
    récompense est tirée d'une loi N(q*(a), 1).
    """

    def __init__(self, k: int = 10, seed: int | None = None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=k)
        self.optimal_action = int(np.argmax(self.q_star))

    def step(self, action: int) -> float:
        """Retourne une récompense stochastique pour l'action choisie."""
        return float(self.rng.normal(loc=self.q_star[action], scale=1.0))

    def reset(self, seed: int | None = None) -> None:
        """Réinitialise les vraies valeurs des bras."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(loc=0.0, scale=1.0, size=self.k)
        self.optimal_action = int(np.argmax(self.q_star))


# ============================================================================
# 2. AGENTS
# ============================================================================


class EpsilonGreedyAgent:
    """Agent ε-greedy avec moyenne empirique des récompenses."""

    def __init__(self, k: int, epsilon: float = 0.1, seed: int | None = None):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N = np.zeros(k, dtype=int)
        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.k))
        return int(np.argmax(self.Q))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class OptimisticAgent:
    """Initialise Q à une valeur optimiste pour forcer l'exploration."""

    def __init__(self, k: int, q_init: float = 5.0, alpha: float = 0.1):
        self.k = k
        self.alpha = alpha
        self.Q = np.full(k, q_init, dtype=float)

    def select_action(self) -> int:
        return int(np.argmax(self.Q))

    def update(self, action: int, reward: float) -> None:
        self.Q[action] += self.alpha * (reward - self.Q[action])


class UCBAgent:
    """Sélection par borne supérieure de confiance."""

    def __init__(self, k: int, c: float = 2.0):
        self.k = k
        self.c = c
        self.Q = np.zeros(k)
        self.N = np.zeros(k, dtype=int)
        self.t = 0

    def select_action(self) -> int:
        self.t += 1
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
        bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(self.Q + bonus))

    def update(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


class ThompsonSamplingAgent:
    """Thompson Sampling avec a priori gaussien (récompenses continues)."""

    def __init__(self, k: int, sigma: float = 1.0, seed: int | None = None):
        self.k = k
        self.sigma = sigma
        self.mu = np.zeros(k)
        self.tau = np.full(k, 1.0 / sigma**2)
        self.rng = np.random.default_rng(seed)

    def select_action(self) -> int:
        sampled = self.rng.normal(loc=self.mu, scale=1.0 / np.sqrt(self.tau))
        return int(np.argmax(sampled))

    def update(self, action: int, reward: float) -> None:
        prior_tau = self.tau[action]
        prior_mu = self.mu[action]
        likelihood_tau = 1.0 / self.sigma**2
        new_tau = prior_tau + likelihood_tau
        new_mu = (prior_tau * prior_mu + likelihood_tau * reward) / new_tau
        self.mu[action] = new_mu
        self.tau[action] = new_tau


class GradientBanditAgent:
    """Apprend des préférences H(a) via softmax + ascension de gradient."""

    def __init__(self, k: int, alpha: float = 0.1, seed: int | None = None):
        self.k = k
        self.alpha = alpha
        self.H = np.zeros(k)
        self.avg_reward = 0.0
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _softmax(self) -> np.ndarray:
        exp_H = np.exp(self.H - np.max(self.H))
        return exp_H / np.sum(exp_H)

    def select_action(self) -> int:
        probs = self._softmax()
        return int(self.rng.choice(self.k, p=probs))

    def update(self, action: int, reward: float) -> None:
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t
        probs = self._softmax()
        baseline = self.avg_reward
        for a in range(self.k):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * probs[a]


# ============================================================================
# 3. EXPÉRIENCE
# ============================================================================


def run_experiment(
    agent_factory,
    n_runs: int = 500,
    n_steps: int = 1000,
    k: int = 10,
):
    """
    Exécute n_runs expériences indépendantes et retourne :
      - rewards : récompense moyenne à chaque pas (moyennée sur runs)
      - optimal_pct : % d'actions optimales à chaque pas (moyenné sur runs)
    """
    rewards = np.zeros(n_steps)
    optimal_pct = np.zeros(n_steps)

    for run in range(n_runs):
        env = BanditEnvironment(k=k, seed=run)
        agent = agent_factory(k)
        for t in range(n_steps):
            a = agent.select_action()
            r = env.step(a)
            agent.update(a, r)
            rewards[t] += r
            if a == env.optimal_action:
                optimal_pct[t] += 1.0

    rewards /= n_runs
    optimal_pct = 100.0 * optimal_pct / n_runs
    return rewards, optimal_pct


# ============================================================================
# 4. POINT D'ENTRÉE
# ============================================================================


def main():
    np.random.seed(42)

    agents = {
        "ε-greedy (ε=0.1)": lambda k: EpsilonGreedyAgent(k, epsilon=0.1, seed=0),
        "ε-greedy (ε=0.01)": lambda k: EpsilonGreedyAgent(k, epsilon=0.01, seed=0),
        "Optimistic (Q₀=5)": lambda k: OptimisticAgent(k, q_init=5.0, alpha=0.1),
        "UCB (c=2)": lambda k: UCBAgent(k, c=2.0),
        "Thompson Sampling": lambda k: ThompsonSamplingAgent(k, seed=0),
        "Gradient (α=0.1)": lambda k: GradientBanditAgent(k, alpha=0.1, seed=0),
    }

    n_runs = 500
    n_steps = 1000
    k = 10

    print(f"Comparaison de {len(agents)} agents sur le {k}-armed testbed")
    print(f"({n_runs} runs indépendants, {n_steps} pas chacun)\n")

    results = {}
    for name, factory in agents.items():
        print(f"  Exécution de {name}...")
        results[name] = run_experiment(factory, n_runs=n_runs, n_steps=n_steps, k=k)

    print("\nGénération des graphiques...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = {
        "ε-greedy (ε=0.1)": "#2563eb",
        "ε-greedy (ε=0.01)": "#0ea5e9",
        "Optimistic (Q₀=5)": "#f59e0b",
        "UCB (c=2)": "#9333ea",
        "Thompson Sampling": "#16a34a",
        "Gradient (α=0.1)": "#dc2626",
    }

    for name, (rewards, _) in results.items():
        axes[0].plot(rewards, label=name, color=colors.get(name), linewidth=1.5)
    axes[0].set_xlabel("Pas de temps")
    axes[0].set_ylabel("Récompense moyenne")
    axes[0].set_title(
        "Comparaison des algorithmes de bandit — Récompense moyenne\n"
        f"({n_runs} runs sur le {k}-armed testbed de Sutton & Barto)"
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    for name, (_, optimal_pct) in results.items():
        axes[1].plot(optimal_pct, label=name, color=colors.get(name), linewidth=1.5)
    axes[1].set_xlabel("Pas de temps")
    axes[1].set_ylabel("% d'actions optimales")
    axes[1].set_title("Comparaison des algorithmes de bandit — Pourcentage d'action optimale")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("bandits_comparison.png", dpi=120, bbox_inches="tight")
    print("✓ Graphique enregistré : bandits_comparison.png")
    plt.show()

    print("\nRésumé des performances finales (moyenne des 100 derniers pas) :")
    print("-" * 65)
    print(f"  {'Agent':<25} {'Récompense':>12} {'% Optimal':>12}")
    print("-" * 65)
    for name, (rewards, optimal_pct) in results.items():
        avg_reward = float(np.mean(rewards[-100:]))
        avg_optimal = float(np.mean(optimal_pct[-100:]))
        print(f"  {name:<25} {avg_reward:>12.3f} {avg_optimal:>11.1f}%")
    print("-" * 65)


if __name__ == "__main__":
    main()
