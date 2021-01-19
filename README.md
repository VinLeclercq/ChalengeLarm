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

![installation_repo](doc/install_git.png "le package challenche_pkg est dans catkin_ws/src")

## Lancer les différents challenges
### Configuration générale

- Lancer la simulation gazebo avec l'environnement adapté (challenge-x)

```bash
roslaunch larm challenge-x.launch
```

- Ouvrez RViz et configurez de sorte à avoir :
    - une Map (topic /map)
    - le RobotModel

### Challenge 1

- Lancer le launch de navigation

```bash
roslaunch challenche_pkg navigation.launch
```

- Utiliser l'outil *2D Nav Goal* sur RViz pour donner une destination **sur une zone grise connue** au robot (la carte se met à jour automatiquement)

### Challenge 2

### Challenge 3