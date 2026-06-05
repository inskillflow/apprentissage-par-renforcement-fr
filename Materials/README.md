# Chapitre 1-bis — Les Problèmes de Bandits

Ce dossier contient le chapitre complet sur les **problèmes de bandits multi-bras** (Multi-Armed Bandits, MAB), forme la plus simple d'apprentissage par renforcement.

## Contenu du dossier

| Fichier | Description |
|---|---|
| `22-Chapitre22-Les-Problemes-de-Bandits.md` | Cours complet (13 sections, quiz, synthèse) |
| `22.1-Quiz-Bandits.ipynb` | Quiz interactif (QCM corrigés + questions de réflexion) — dossier `Materials/` |
| `22.2-Revision-et-coder-Bandits.ipynb` | Notebook : coder et comparer les 6 agents (10-armed testbed) — dossier `Materials/` |
| `22.3-bandits.py` | Implémentation Python exécutable des 6 algorithmes de bandits — dossier `Materials/` |
| `22.4-Annexe-Comprendre-les-Bandits.ipynb` | Annexe pédagogique (vulgarisation) : `|S|=1`, formule `Q_t(a)`, `Q(a)` vs `Q(s,a)`, MSN.com / bandits contextuels, tests A/B, métaphores des 4 algorithmes — dossier `Materials/` |
| `22.5-Annexe-Comprendre-les-Bandits.md` | Même annexe en version Markdown (lecture directe, sans notebook) — dossier `Materials/` |
| `requirements.txt` | Dépendances Python (numpy, matplotlib) |
| `README.md` | Ce fichier |

## Plan du chapitre

1. **Vue d'ensemble** — Qu'est-ce qu'un problème de bandit ?
2. **Le bandit multi-bras** — Définition formelle (k-armed bandit)
3. **Applications industrielles** des bandits simples
4. **Algorithmes Value-Based** — ε-greedy, Optimistic Init, UCB, Thompson Sampling
5. **Algorithmes par gradients** — Gradient Bandit
6. **Comparaison directe** des algorithmes
7. **Paramètres vs Hyperparamètres**
8. **Implémentation Python complète et exécutable**
9. **Bandits complexes** — Combinatoires et Contextuels
10. **Applications industrielles** des bandits complexes
11. **Défis et limites** des bandits complexes
12. **Quiz** (15 questions avec solutions repliables)
13. **Synthèse du chapitre**

## Lancer la démonstration Python

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Exécution

```bash
python 22.3-bandits.py
```

Cela génère :

- Un fichier `bandits_comparison.png` avec deux graphiques (récompense moyenne et % d'action optimale)
- Un tableau récapitulatif des performances finales dans la console

### Exemple de sortie console

```
Comparaison de 6 agents sur le 10-armed testbed
(500 runs indépendants, 1000 pas chacun)

  Exécution de ε-greedy (ε=0.1)...
  Exécution de ε-greedy (ε=0.01)...
  Exécution de Optimistic (Q₀=5)...
  Exécution de UCB (c=2)...
  Exécution de Thompson Sampling...
  Exécution de Gradient (α=0.1)...

Résumé des performances finales (moyenne des 100 derniers pas) :
-----------------------------------------------------------------
  Agent                       Récompense    % Optimal
-----------------------------------------------------------------
  ε-greedy (ε=0.1)                 1.547        78.3%
  ε-greedy (ε=0.01)                1.490        72.1%
  Optimistic (Q₀=5)                1.612        85.4%
  UCB (c=2)                        1.687        89.2%
  Thompson Sampling                1.701        90.1%
  Gradient (α=0.1)                 1.554        78.8%
-----------------------------------------------------------------
```

> **Note :** les chiffres exacts varient selon la seed aléatoire, mais l'ordre relatif (UCB ≈ Thompson > Optimistic > Gradient ≈ ε-greedy) est robuste.

## Pré-requis pédagogiques

- Avoir lu le **Chapitre 1** (Introduction au RL)
- Notions de base en Python (numpy)
- Notions de probabilités (espérance, distribution gaussienne)

## Chapitres suivants

Le chapitre 1-bis prépare directement à :

- **Chapitres 4-5** (MDP) — généralisation du bandit aux problèmes multi-états
- **Chapitre 8** (Value-Based vs Policy-Based) — les bandits ont les deux familles à l'état pur
- **P12** (Q-Learning) — Q-Learning généralise les mises à jour de bandit aux MDP
- **P19** (PPO) — extension Deep RL du Gradient Bandit

## Références

- **Sutton & Barto** — *Reinforcement Learning : An Introduction* (2018), Chapitre 2
- **Lattimore & Szepesvári** — *Bandit Algorithms* (2020) — disponible gratuitement en PDF
- **Li et al., 2010** — *A Contextual-Bandit Approach to Personalized News Article Recommendation* (Yahoo!)

---

*Cours créé par Dr. Haythem REHOUMA — Apprentissage par Renforcement*
