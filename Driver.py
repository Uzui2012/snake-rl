from IBP.IBP import IBP
from Board.Board import Board, BOARD_HEIGHT, BOARD_WIDTH
from DQN.DQN_agent import DQN_agent
from gym_snake.gym_snake.envs.constants import GridType
from gym_snake.gym_snake.envs.snake_env import SnakeEnv
PC_PATH = "C:\\Users\\killi\Documents\\Repositories\\snake-rl\\"
LAPTOP_PATH = "C:\\Users\\killi\\Repos\\snake-rl\\"

CUDA_FLAG = True

class Snake_5x5_DeadApple(SnakeEnv):
    def __init__(self):
        super().__init__(grid_size=5, initial_snake_size=2) 

if __name__ == "__main__":
    try:     
        gym = Snake_5x5_DeadApple()

        dqn_agent = DQN_agent(action_number=4,
                              frames=1, 
                              learning_rate=0.0001,
                              discount_factor=0.99, 
                              batch_size=8,
                              epsilon=1,
                              save_model=False,
                              load_model=False,
                              path=PC_PATH
                              +"DQN_trained_model\\10x10_model_with_tail.pt",
                              epsilon_speed=1e-4,
                              cuda_flag=CUDA_FLAG)

        board = Board(BOARD_HEIGHT, BOARD_WIDTH, dqn_agent=dqn_agent)

        ibp = IBP(environment=board, 
                  proj_path=PC_PATH, 
                  cuda_flag=CUDA_FLAG)

        num_eps = 1000
        scores = []
        scores = ibp.run(board, num_eps)
            
        ibp.plot_results(scores)
        pass
    except BaseException as error:
        print('An exception occurred: {}'.format(error))