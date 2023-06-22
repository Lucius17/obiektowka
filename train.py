
import pygame
import sys
import random
import numpy as np
import os.path
# np.set_printoptions(threshold=sys.maxsize)

pygame.init()
pygame.font.init()
myFont = pygame.font.SysFont("monospace", 35)

screen = pygame.display.set_mode((450, 800))

background = pygame.image.load("bg.png")
background = pygame.transform.scale(background, (450, 800))
img_pipe = pygame.image.load("pipe.png")
img_pipe = pygame.transform.scale(img_pipe, (100, 800))
clock = pygame.time.Clock()
pipe_speed = 5
pipe_gap = 220
pipe_list = []
img_bird = pygame.image.load("bird.png")
img_bird = pygame.transform.scale(img_bird, (64, 48))
bird_color = pygame.image.load("bird_color.png").convert_alpha()
bird_color = pygame.transform.scale(bird_color, (64, 48))

x, y = 30, 40
FPS = 60
bird_amount = 30
epsilon = 0.99
discount_factor = 0.8
learning_rate = 0.9
highscore = 0

# print(type(np.zeros((1, 1))))


def changColor(image, hue):
    color = pygame.Color(0)
    color.hsla = (hue, 100, 50, 100)
    colouredImage = pygame.Surface(image.get_size())
    colouredImage.fill(color)

    finalImage = image.copy()
    finalImage.blit(colouredImage, (0, 0), special_flags=pygame.BLEND_MULT)
    return finalImage


def get_next_action(x, y, lower, epsilon, q_values):
    if np.random.random() < epsilon:
        t = q_values[x][y][lower]
        np.argmax(t)
        ind = np.unravel_index(np.argmax(t, axis=None), t.shape)

        return ind[0]

    else:  # choose a random action
        return np.random.randint(2)


class Bird:
    def __init__(self, q_values=np.zeros((x, y, 2, 2))):
        self.q_values = q_values
        self.x = 100
        self.y = 400
        self.color = changColor(bird_color, random.randint(1, 360))
        self.fly = False
        self.vel_y = 10
        self.dead = False
        self.hit_box = pygame.Rect(self.x, self.y, 64, 64)
        self.score = 0
        self.reward = 0
        self.horizontal_dif = 0
        self.height_dif = 0
        self.collected = False
        self.lower = 0

    def move(self):

        # gravity
        self.vel_y += 0.7
        if self.vel_y > 5:
            self.vel_y = 5
        if self.y < 800 and self.y > 0:
            self.y += int(self.vel_y)
        else:
            self.dead = True
            self.reward = -1000
        if self.fly and not self.dead and self.y > 0:
            self.vel_y = -10
            self.fly = False
        self.hit_box = pygame.Rect(self.x, self.y, 64, 64)
        if len(pipe_list) > 0:
            if self.y > pipe_list[0].y:
                self.lower = 1
            else:
                self.lower = 0
            self.horizontal_dif = abs(
                round((self.x+32 - pipe_list[0].x-100)/20))
            self.height_dif = abs(
                round((self.y+32 - pipe_list[0].y)/20))
            if self.hit_box.colliderect(pipe_list[0].hit_box_down) or self.hit_box.colliderect(pipe_list[0].hit_box_up):
                self.reward = -100
                self.dead = True
            if self.hit_box.colliderect(pipe_list[0].hit_box_coin) and not self.collected:
                self.collected = True
                self.reward = 1000
                self.score += 1
        else:
            self.lower = 0

    def draw(self):
        screen.blit(img_bird, (self.x, self.y))
        screen.blit(self.color, (self.x, self.y))
        if len(pipe_list) > 0:
            if self.lower == 1:
                pygame.draw.line(screen, (255, 0, 0), (self.x+32, self.y+32),
                                 (self.x+32, pipe_list[0].y), 3)
            else:
                pygame.draw.line(screen, (0, 255, 0), (self.x+32, self.y+32),
                                 (self.x+32, pipe_list[0].y), 3)
                pygame.draw.line(screen, (255, 0, 0), (self.x+32, pipe_list[0].y),
                                 (pipe_list[0].x+100, pipe_list[0].y), 3)


class Pipe:
    def __init__(self):
        self.x = 500
        self.y = random.randint(300, 600)
        self.hit_box_down = pygame.Rect(
            self.x, self.y, img_pipe.get_width(), img_pipe.get_height())
        self.hit_box_up = pygame.Rect(
            self.x, self.y - pipe_gap - img_pipe.get_height(), img_pipe.get_width(), img_pipe.get_height())
        self.hit_box_coin = pygame.Rect(
            self.x+100, self.y-pipe_gap, 2, pipe_gap)

    def draw(self):
        screen.blit(img_pipe, (self.x, self.y))
        screen.blit(pygame.transform.flip(
            img_pipe, False, True), (self.x, self.y - pipe_gap - img_pipe.get_height()))

        pygame.draw.rect(screen, (250, 0, 0), self.hit_box_down, 2)
        pygame.draw.rect(screen, (250, 0, 0), self.hit_box_up, 2)
        pygame.draw.rect(screen, (250, 255, 0), self.hit_box_coin, 2)

    def move(self):
        self.x -= pipe_speed
        self.hit_box_down.move_ip(-pipe_speed, 0)
        self.hit_box_up.move_ip(-pipe_speed, 0)
        self.hit_box_coin.move_ip(-pipe_speed, 0)


if os.path.exists("model.npy"):
    loaded_arr = np.load('model.npy')
    bird_list = [Bird(loaded_arr) for i in range(bird_amount)]
    super_bird=Bird(loaded_arr)

else:
    bird_list = [Bird() for i in range(bird_amount)]
    super_bird=Bird()
# bird_list = [Bird() for i in range(bird_amount)]


dead_birds = []
episodes = 0
pipe_list.append(Pipe())
while True:
    screen.blit(background, (0, 0))

    for pipe in pipe_list:
        pipe.draw()
        pipe.move()
        if pipe.x < -150:
            for bird in bird_list:
                bird.collected = False
            pipe_list.remove(pipe)
            pipe_list.append(Pipe())

    for event in pygame.event.get():  # event handling
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                FPS += 15
            if event.key == pygame.K_LEFT and FPS > 15:
                FPS -= 15

    for bird in bird_list:
        if not bird.dead:

            action_index = int(get_next_action(
                bird.horizontal_dif, bird.height_dif, bird.lower, epsilon, bird.q_values))
            if action_index == 1:
                bird.fly = True
                # store the old row and column indexes
            old_horizontal_dif, old_height_dif, old_lower = bird.horizontal_dif, bird.height_dif, bird.lower

            bird.move()
            bird.draw()

            reward = bird.reward
            if action_index == 1:
                reward -= 0.5
            if not bird.dead:
                reward += 1
            if bird.lower == 1 and bird.height_dif > old_height_dif:
                reward -= 5

            bird.reward = 0
            old_q_value = bird.q_values[old_horizontal_dif,
                                        old_height_dif, old_lower, action_index]

            temporal_difference = reward + (discount_factor * np.max(
                bird.q_values[old_horizontal_dif,
                              old_height_dif, old_lower])) - old_q_value
            new_q_value = old_q_value + (learning_rate * temporal_difference)

            bird.q_values[old_horizontal_dif, old_height_dif,
                          old_lower, action_index] = new_q_value

            if bird.score > highscore:
                highscore = bird.score
                super_bird=bird.q_values
                np.save('model.npy', super_bird)
            screen.blit(myFont.render(
                (f'{bird.score}'), True, (255, 255, 255)), (200, 60))
        else:
            if len(bird_list) > 1:
                dead_birds.append(bird.q_values)
                bird_list.remove(bird)
            else:
                episodes += 1
                # super_bird = bird_list[0].q_values
                bird_list = [Bird(super_bird)
                             for i in range(bird_amount)]
                pipe_list = [Pipe()]
                dead_birds = []
    screen.blit(myFont.render(
        (f'Episodes: {episodes}'), True, (255, 255, 255)), (0, 0))
    screen.blit(myFont.render(
        (f'Highscore: {highscore}'), True, (255, 255, 255)), (0, 30))
    pygame.display.update()
    clock.tick(FPS)
