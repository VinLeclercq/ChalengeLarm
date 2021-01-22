# ChallengeLarm

**Groupe n°14**
## Membres :
- Vinciane LECLERCQ
- Enzo BERNARD
- Erwan MERLY

## Installation
1. Créer un Rosject ROS-Kinetic
2. Penser à cloner le dossier `simulation_ws`

```bash
rm -fr simulation_ws
git clone https://github.com/ceri-num/LARM-RDS-Simulation-WS.git simulation_ws
cd simulation_ws
catkin_make
source devel/setup.bash
```

3. Cloner ce repository git à la place du dossier `catkin_ws` actuel

```bash
rm -fr catkin_ws
git clone https://github.com/VinLeclercq/ChallengeLarm.git catkin_ws
cd catkin_ws
catkin_make
source devel/setup.bash
```

Vous devriez avoir le package `challenche_pkg` dans votre dossier `catkin_ws`

## Lancer les différents challenges
### Configuration générale

- Lancer la simulation gazebo avec l'environnement adapté (challenge-x)

```bash
roslaunch larm challenge-x.launch
```

- Ouvrez RViz et charger la configuration préfaite. Elle se trouve ici : ```catkin_ws/src/rviz/challenge_config.rviz```
Elle contient :
    - une map
    - le modèle du robot
    - le scanner pour mieux voir l'orientation du robot
    - les markers pour le challenge 3

### Challenge 1

- Lancer le launch de navigation

```bash
roslaunch challenche_pkg navigation.launch
```

- Utiliser l'outil *2D Nav Goal* sur RViz pour donner une destination **sur une zone grise connue** au robot (la carte se met à jour automatiquement)

### Challenge 2

- Dans une nouvelle console rentrer ```rostopic echo /bottle```

- Lancer le launch de mapping

```bash
roslaunch challenche_pkg mapping.launch
```

- Utiliser l'outil *2D Nav Goal* sur RViz pour donner une destination **sur une zone grise connue** au robot (la carte se met à jour automatiquement)

- Les positions des cannettes sont envoyées sur le topic ```/bottle``` et visible dans la console préalablement configurée

### Challenge 3

- Lancer le launch de exploration

```bash
roslaunch challenche_pkg exploration.launch
```

- Dans une nouvelle console rentrer ```rostopic echo /bottle```

- Utiliser l'outil *Publish Point* pour placer plusieurs points de sorte à avoir un polygone **fermé** et réutiliser l'outil dans ce polygone **sur une zone grise connue**. Le robot devrait se mettre à explorer la zone de manière autonome.

- Les positions des cannettes sont envoyées sur le topic ```/bottle``` et visible dans la console préalablement configurée

## Difficultés rencontrées

- Le scan du robot ne permet de "voir" les canettes, il est donc plus compliqué pour nous d'obtenir le positionnement exact.
- L'utilisation et le paramétrage de *frontier_exploration* a été complexe. En effet, le robot ne parvient pas à tout explorer de manière autonome et se perd souvent dans des murs. Nous pensons que cela provient sûrement du rafraîchissement de la map qui n'est pas suffisamment élevé mais n'avons pas trouvé de solution.
- Lors de la localisation de cannettes, si le robot de voit pas de cannette il retourne un message d'erreur dans la console. La grosse partie en rouge peut ne pas s'afficher si on place un try-exept dans le code. Néanmoins nous n'avons pas réussi a affacer complètement l'erreur dans la console 

## Conclusion

Cette UV était une première expérience en robotique pour nous, et ce fut un mois enrichissant. On regrette juste de ne pas avoir pu toucher à de vrais robots et pouvoir voir la magie opérer devant nos yeux.
