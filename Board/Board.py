import os
import sys
sys.path.append("..")
import pickle
import time
import torch
from scipy import ndimage, misc
import numpy as np
import pygame
import matplotlib.pyplot as plt
from DQN.DQN_agent import DQN_agent
from apple import apple
from collision_handler import collision
from games_manager import games_manager
from snake import Snake
import skimage as skimage
from skimage import transform, color, exposure
import cv2
from skimage.transform import rotate
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

snake_starting_pos = (1, 3)
# ( bool (do we want to load) ,  filename )
load_tables_from_file = (False, "9x9model")
range_of_apple_spawn = (1, 2)

BOARD_WIDTH = 5
BOARD_HEIGHT = 5

class Board():
    def __init__(self, height, width, dqn_agent):
        self.clockobject = pygame.time.Clock()
        self.was_apple_last = False
        pygame.init()
        self.block_size = 50
        self.height = height
        self.width = width
        self.screen = pygame.display
        self.screen.set_mode([self.block_size * self.height,
                             self.block_size * self.width])
        self.snake = Snake(self.block_size, self.screen, snake_starting_pos)
        self.running = True
        self.apple = apple(height, width, self.block_size, self.screen,
                            range_of_apple_spawn, self.snake)
        self.new_pos_x = 0
        self.new_pos_y = 0
        self.index = 96662 ### Why 77000 and not 0?
        self.index_helper = 0
        self.collision = collision(self.apple, self.snake)
        self.apple.spawn_apple()
        self.tick = 0
        self.game_manager = games_manager((self.apple.x, self.apple.y), 
                                          snake_starting_pos,
                                          self.height,
                                          self.width,
                                          self.snake, load_tables_from_file)
        self.current_game = self.game_manager.switch_state_table(self.apple.apple_position, self.snake.snake_head,
                                                                 snake_starting_pos)
        self.decide_epsilon_greedy()
        self.games_count = 0
        self.longest_streak = 0

        self.dqn_agent = dqn_agent
        self.reward = 0
        self.action = None
        self.speed = 9000
        self.debug = []
        self.previous_gan_action = None
        self.temporal_frame_list = []
        self.past_action = None
        self.apple_movement_flag = False

    def decide_epsilon_greedy(self):
        if load_tables_from_file[0]:
            self.epsilon = 0
        else:
            self.epsilon = 0.1

    def run(self):
        num_images = 0
        score_arr = []
        episodes = 2000
        for i in range(0, episodes):
            score = 0
            print(f"New Episode: {i}")
            while self.running:
                
                pygame.display.flip()
                reward = self.collision.return_reward(self.height, self.width)
                self.clockobject.tick(self.speed)
                
                
                #print(f"Reward this step: {reward}")
                """
                self.process_input()
                self.draw_sprites()

                self.snake.draw_segment()
                self.apple.draw_apple()
                img = self.get_state()

                # plt.imshow(img, cmap='gray', vmin=0, vmax=1)
                # plt.show()
                """
                img = self.get_state()
                action = self.dqn_agent.make_action(img, reward,
                        True if reward == -1 or reward == 10 else False)


                snake.run_step(action, apple_crawl=False)
                #action = int(input())
                #print(action)
                ### TEMP RUN AT HUMAN SPEED
                #time.sleep(1/5)
                # self.snake.action(action)
                # for moving apple
                """
                if self.apple_movement_flag == False:

                    # was if reward == 10 or

                    if reward != 10:
                        self.snake.action(action)
                    if reward == 10:
                        self.apple_movement_flag = True
                        action = None
                        self.past_action = action
                        self.new_pos_x, self.new_pos_y = self.apple.spawn_apple()
                
                # self.snake.action(action)
                self.tick += 1
                self.lose_win_scenario(reward)
                self.games_count += 1

                if self.apple_movement_flag == False:
                    self.snake.move_segmentation()
                # for moving apple
                if self.apple_movement_flag:
                    action = None
                    if self.apple.x == self.new_pos_x and self.apple.y == self.new_pos_y:
                        self.apple_movement_flag = False

                    else:
                        self.apple.move_apple(curr_apple_pos=(self.apple.x, self.apple.y),
                                            target_pos=(self.new_pos_x, self.new_pos_y))
                """
                
                self.create_actions_channels(action, img, reward)
                num_images += 1

                if reward == 10:
                    score += 1
                if reward == -1:
                    break;
            score_arr.append(score)   
            if i%100 == 0:
                plot_results(score_arr) 
        print(num_images)
        
        os.remove(f'D:/ProjPickleDump/train_steps/S_images/state_s_{self.index + num_images}.pickle')

    def draw_sprites(self):
        self.draw_board()

        self.snake.draw_snake()

    def lose_win_scenario(self, reward):
        if reward == -1 or reward == 10:
            if reward == -1:
                print("Apple score {}".format(self.longest_streak))
                self.longest_streak = 0
                self.snake.reset_snake()
            if reward == 10:
                self.longest_streak += 1
                if self.longest_streak == 110:
                    self.epsilon = 0

            # print(reward)

            # Change me if you want random apple

    def draw_board(self):
        for y in range(self.height):
            for x in range(self.width):
                # Rectangle drawing
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen.get_surface(), (0, 0, 0), rect)

    def get_state(self):

        # I take red blue green RGB
        red = pygame.surfarray.pixels_red(self.screen.get_surface())
        green = pygame.surfarray.pixels_green(self.screen.get_surface())
        blue = pygame.surfarray.pixels_blue(self.screen.get_surface())

        inside = np.array([red, green, blue]).transpose((2, 1, 0))
        # plt.imshow(inside)
        # plt.show()
        return self.ProcessGameImage(inside)

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Be interpreter friendly
                self.game_manager.save_model(f"{BOARD_WIDTH}x{BOARD_HEIGHT}model")
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                # print("hgere")
                action = None
                if event.key == pygame.K_RIGHT:
                    pass
                if event.key == pygame.K_UP:
                    self.speed = 9000

                if event.key == pygame.K_LEFT:
                    # print("kkk")
                    action = 0
                if event.key == pygame.K_DOWN:
                    self.speed = 1
                # print(action)
            pygame.display.update()
        return

    def ProcessGameImage(self, RawImage):
        GreyImage = skimage.color.rgb2gray(RawImage)
        # Get rid of bottom Score line
        # Now the Pygame seems to have turned the Image sideways so remove X direction
        # RawImage = GreyImage[0:400, 0:400]
        # plt.imshow(GreyImage)
        # plt.show()
        # print("Cropped Image Shape: ",CroppedImage.shape)

        # ReducedImage = skimage.transform.resize(RawImage, (84, 84,3), mode='reflect')
        # ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range=(0, 255))
        img = cv2.resize(GreyImage, (84, 84))

        # plt.imshow(img.astype(np.uint8))
        # plt.show()
        # print(img.shape)
        # img = ndimage.rotate(img, 270, reshape=False)
        # if self.index > 0:
        #     plt.imshow(img)
        #     plt.savefig(f"S'_images/{self.index-1}.png")
        #     plt.close()
        #
        #
        #
        # plt.imshow(img)
        # plt.savefig(f"C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\train\\S_images\\{self.index}.png")
        # plt.close()
        # self.index += 1

        return img

    def create_actions_channels(self, action, img, reward):

        # plt.imshow(img,cmap='gray',vmax=1,vmin=0)
        # plt.show()
        if action == None:
            action = np.zeros(4)
            self.was_apple_last = True
        else:
            action_vec = np.zeros(4)
            action_vec[action] = 1
            action = action_vec

        np_reward = np.zeros(3)
        # mapping of reward 0 => 10 , 1 => -1 ,  2 => -0.1
        if reward == 10:
            np_reward[0] = 1

        elif reward == -1:
            np_reward[1] = 1

        else:
            np_reward[2] = 1

        np_reward = torch.from_numpy(np_reward)

        # print("reward {} vector {}".format(reward, np_reward))
        action2 = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * 0
        action = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * torch.from_numpy(action) \
            .unsqueeze(1) \
            .unsqueeze(2)
        # noise = torch.randn(1, 84, 84)
        state_action = torch.cat([torch.from_numpy(img).unsqueeze(0), action], dim=0)
        # state_action = torch.cat([noise.double(), action], dim=0)
        # state = torch.from_numpy(img)
        # if self.index > 0:
        #     with open(f"train_reward/future/state_s_{self.index - 1}.pickle", 'wb') as handle:
        #
        #         pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'train_reward/now/state_s_{self.index}.pickle', 'wb') as handle:
        #     pickle.dump((state, np_reward), handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #
        # # GAN
        if self.index > 0:
            with open(f"D:/ProjPickleDump/train_steps/Sa_images/state_s_{self.index - 1}.pickle", 'wb') as handle:
                future_state = torch.from_numpy(img).unsqueeze(0)
                pickle.dump(future_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'D:/ProjPickleDump/train_steps/S_images/state_s_{self.index}.pickle', 'wb') as handle:
            pickle.dump((state_action, np_reward), handle, protocol=pickle.HIGHEST_PROTOCOL)

        # RNN
        # if len(self.temporal_frame_list)==5:
        #     # #debug
        #     # for img in self.temporal_frame_list:
        #     #     plt.imshow(img.squeeze())
        #     #     plt.show()
        #
        #     with open(f'train/video/state_s_{self.index}.pickle', 'wb') as handle:
        #         video = np.swapaxes(torch.stack(self.temporal_frame_list[:4]),0,1)
        #         ground_truth = self.temporal_frame_list[4]
        #         pickle.dump((video,ground_truth), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.index += 1
        #
        #     self.temporal_frame_list.pop(0)
        # self.temporal_frame_list.append(torch.from_numpy(img).unsqueeze(0))
        # RNN end
        # generate validation images
        # with open(f'validate_gan/state_s_{self.index}.pickle', 'wb') as handle:
        #  pickle.dump((state_action,reward), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.previous_gan_action = action

    def render(self):
        self.draw_sprites()
        self.snake.draw_segment()
        self.apple.draw_apple()
        return 

    def update_game(self, user_input, crawl_flag=False):
        reward, done = self.get_reward()
        img = self.get_state()
        if(user_input is None):
            action = self.dqn_agent.make_action(img, reward,
                      True if reward == -1 or reward == 10 else False)
        else:
            action = user_input
        if reward != 10:
            self.snake.action(action)
        if reward == 10:
            action = None
            self.past_action = action
            self.new_pos_x, self.new_pos_y = self.apple.spawn_apple()
            self.apple.move_apple(curr_apple_pos=(self.apple.x, self.apple.y),
                                        target_pos=(self.new_pos_x, self.new_pos_y),
                                        crawl_flag=crawl_flag)
        self.snake.move_segmentation()
        
        self.tick += 1
        self.lose_win_scenario(reward)
        self.games_count += 1

        return img, reward, done

    def get_reward(self):
        reward = self.collision.return_reward(self.height, self.width)
        self.clockobject.tick(self.speed)
        done = False
        if reward == -1:
            done = True
        return reward, done

    def run_step(self, user_input=None, apple_crawl=True):
        pygame.display.flip()
        self.update_game(user_input, crawl_flag=apple_crawl)
        self.process_input()
        self.render()
        user_input = None
        rwd, done = self.get_reward()
        return (self.get_state(), rwd, done)

    def get_GAN_guess(self):
        pass

def plot_results(rewards):
    plt.figure(figsize=(12,5))
    plt.title("Rewards")
    plt.plot(rewards, alpha=0.6, color='red')
    plt.savefig("Snake_Rewards_plot.png")
    plt.close()

if __name__ == "__main__":
    try:
        snake = Board(BOARD_WIDTH, BOARD_HEIGHT)
        #snake.run()
        while True:
            user_input = int(input())
            pygame.display.set_caption(str(user_input))
            snake.run_step(user_input, apple_crawl=False)
            snake.run_step(22)
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
    #snake.render()
    