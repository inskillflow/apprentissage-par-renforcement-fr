# Propositions de Projets

# **1. LunarLander (Facile à Moyen)**  
**Description :**  
L'agent doit contrôler un module lunaire et le poser en toute sécurité sur une plateforme en ajustant sa poussée et son inclinaison.  

**Objectif :**  
- Minimiser le carburant utilisé.  
- Éviter les collisions.  
- Atterrir au centre de la plateforme.  

**Pourquoi c'est intéressant ?**  
- Il introduit des actions continues et une gestion de ressources (carburant).  
- Les états incluent des paramètres physiques comme la vitesse et la position.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("LunarLander-v2")`  
**Algorithme recommandé :** PPO ou DDPG  

<br/>

# **2. MountainCar (Facile)**  
**Description :**  
Un agent doit pousser une voiture hors d'une vallée en utilisant sa propre inertie pour atteindre le sommet.  

**Objectif :**  
- Apprendre à coordonner les mouvements pour atteindre l’objectif en un minimum d’étapes.  

**Pourquoi c'est intéressant ?**  
- C’est une bonne introduction à la gestion des actions pour maximiser les récompenses différées.  
- Les états incluent la position et la vitesse de la voiture.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("MountainCar-v0")`  
**Algorithme recommandé :** Q-Learning ou PPO  

<br/>

# **3. Flappy Bird (Moyen)**  
**Description :**  
Créer un agent qui apprend à jouer au célèbre jeu Flappy Bird, où l'oiseau doit traverser des obstacles sans tomber ou heurter des tuyaux.  

**Objectif :**  
- Maximiser le score en passant à travers autant de tuyaux que possible.  

**Pourquoi c'est intéressant ?**  
- Introduit des environnements basés sur des images (observations visuelles).  
- Permet de travailler avec des réseaux convolutifs (CNN).  

**Environnement recommandé :**  
- Utiliser `gym-flappy-bird` ou recréer un environnement simple avec PyGame.  
**Algorithme recommandé :** DQN avec CNN  

<br/>

# **4. Snake Game (Moyen)**  
**Description :**  
Créer un agent capable de jouer au jeu Snake, en maximisant la longueur du serpent tout en évitant de se heurter à lui-même ou aux murs.  

**Objectif :**  
- Maximiser le score en mangeant de la nourriture.  

**Pourquoi c'est intéressant ?**  
- Combine planification à court terme et long terme.  
- Introduit des défis comme la gestion de l’état interne (longueur du serpent).  

**Environnement recommandé :**  
- Créez un jeu Snake avec PyGame ou utilisez des packages existants.  
**Algorithme recommandé :** DQN  

<br/>

# **5. Taxi-v3 (Facile)**  
**Description :**  
Un taxi doit prendre et déposer des passagers à des emplacements désignés dans une grille, en minimisant le nombre de mouvements inutiles.  

**Objectif :**  
- Minimiser la distance parcourue pour ramasser et déposer les passagers.  

**Pourquoi c'est intéressant ?**  
- Facile à comprendre et très visuel (représentation en grille).  
- Idéal pour introduire Q-Learning et SARSA.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("Taxi-v3")`  
**Algorithme recommandé :** Q-Learning  

<br/>

# **6. FrozenLake (Facile)**  
**Description :**  
Un agent doit traverser un lac gelé, représenté par une grille, sans tomber dans un trou. Les cases sont glissantes, ce qui ajoute un élément d'incertitude.  

**Objectif :**  
- Trouver le chemin optimal pour atteindre l'objectif tout en évitant les trous.  

**Pourquoi c'est intéressant ?**  
- Introduit des concepts comme l'exploration/exploitation et la gestion des probabilités.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("FrozenLake-v1")`  
**Algorithme recommandé :** Q-Learning  

<br/>

# **7. Pong (Moyen à Avancé)**  
**Description :**  
Créer un agent qui apprend à jouer à Pong, un jeu classique où deux raquettes s'affrontent pour renvoyer une balle.  

**Objectif :**  
- Maximiser le score en battant l’adversaire.  

**Pourquoi c'est intéressant ?**  
- Bon pour introduire des environnements avec des observations visuelles et des décisions rapides.  
- Permet d’appliquer des techniques avancées comme DQN ou PPO avec CNN.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("PongNoFrameskip-v4")`  
**Algorithme recommandé :** DQN ou PPO  


<br/>

# **8. BipedalWalker (Avancé)**  
**Description :**  
Un robot à deux jambes doit apprendre à marcher sur un terrain accidenté.  

**Objectif :**  
- Optimiser les mouvements pour traverser le terrain sans tomber.  

**Pourquoi c'est intéressant ?**  
- Environnement complexe avec des actions continues.  
- Développe une compréhension des algorithmes de contrôle motorisé.  

**Bibliothèque :** OpenAI Gym  
**Environnement :** `gym.make("BipedalWalker-v3")`  
**Algorithme recommandé :** PPO ou DDPG  

<br/>

# **9. Projets sur mesure (Facile):**

  

1. **Labyrinthe (Maze Solver)** :  
   - Créez un labyrinthe où l'agent doit trouver le chemin vers la sortie.  
   - Introduisez des pièges ou des raccourcis.  
   - Algorithme recommandé : Q-Learning.  

2. **Robot Cleaner** :  
   - Simulez un robot aspirateur qui doit nettoyer toutes les cases d'une pièce en minimisant les mouvements.  
   - Algorithme recommandé : SARSA ou Q-Learning.  

3. **Jeu de Tic-Tac-Toe** :  
   - Créez un agent capable de jouer au Tic-Tac-Toe contre un humain ou un autre agent.  
   - Algorithme recommandé : Minimax ou Q-Learning.  
