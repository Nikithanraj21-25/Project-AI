from snake.game import Game, GameConf, GameMode

greedy = "GreedySolver"
hamilton = "HamiltonSolver"

normal = GameMode.NORMAL

conf = GameConf()
conf.solver_name = hamilton
conf.mode = normal
print("Solver: %s" % (conf.mode))
Game(conf).run()
