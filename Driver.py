from IBP.IBP import IBP
from Board.Board import Board, BOARD_HEIGHT, BOARD_WIDTH
from DQN.DQN_agent import DQN_agent




if __name__ == "__main__":
    try:      
        dqn_agent = DQN_agent(action_number=4,
                                        frames=1, 
                                        learning_rate=0.0001,
                                        discount_factor=0.99, 
                                        batch_size=8,
                                        epsilon=1,
                                        save_model=False,
                                        load_model=True,
                                        path="C:\\Users\\killi\\Documents\\Repositories\\snake-rl\\DQN_trained_model\\10x10_model_with_tail.pt",
                                        epsilon_speed=1e-4)

        board = Board(BOARD_HEIGHT, BOARD_WIDTH, dqn_agent=dqn_agent)
        ibp = IBP(dqn_agent=dqn_agent)
        num_eps = 1000
        for ep in range(num_eps):            
            ibp.run(board)
        pass
    except BaseException as error:
        print('An exception occurred: {}'.format(error))