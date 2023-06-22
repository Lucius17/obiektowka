
import pygame
import sys
import random
import numpy as np
import os.path
# np.set_printoptions(threshold=sys.maxsize)
ez_mode=False
pygame.init()
pygame.font.init()
myFont = pygame.font.SysFont("monospace", 35)
myFont1 = pygame.font.SysFont("monospace", 55)

screen = pygame.display.set_mode((450, 800))

game_over= False

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
debug = False
FPS = 60

x, y = 30, 40

bird_amount = 1
epsilon = 1
discount_factor = 0.8
learning_rate = 0.9
highscore = 0

win=False
lose=False


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

    else:
        return np.random.randint(2)


class Player:
    def __init__(self):
        self.x = 100
        self.y = 400
        self.color = changColor(bird_color, 100)
        self.fly = False
        self.vel_y = 10
        self.dead = False
        self.hit_box = pygame.Rect(self.x, self.y, 64, 64)
        self.score = 0
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
                
                self.dead = True
            if self.hit_box.colliderect(pipe_list[0].hit_box_coin) and not self.collected:
                self.collected = True
                
                self.score += 1
        else:
            self.lower = 0

    def draw(self):
        screen.blit(img_bird, (self.x, self.y))
        screen.blit(self.color, (self.x, self.y))


class Bird:
    def __init__(self, q_values=np.zeros((x, y, 2, 2))):
        self.q_values = q_values
        self.x = 100
        self.y = 400
        self.color = changColor(bird_color, 20)
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
                self.reward = 100
                self.score += 1
        else:
            self.lower = 0

    def draw(self):
        screen.blit(img_bird, (self.x, self.y))
        screen.blit(self.color, (self.x, self.y))
        if len(pipe_list) > 0:

            if debug:
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

        if debug:
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

else:
    bird_list = [Bird() for i in range(bird_amount)]


dead_birds = []
episodes = 0
pipe_list.append(Pipe())
player = Player()


while True:
    screen.blit(background, (0, 0))

    for pipe in pipe_list:
        pipe.draw()
        pipe.move()
        if pipe.x < -150:
            player.collected = False
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
            if event.key == pygame.K_SPACE and game_over==False:
                player.fly=True
            if event.key == pygame.K_SPACE and game_over==True:
                game_over=False
                lose=False
                win=False
                player = Player()
                episodes += 1

                super_bird = bird_list[0].q_values
                bird_list = [Bird(super_bird)
                             for i in range(bird_amount)]
                pipe_list = [Pipe()]
                dead_birds = []
            if event.key == pygame.K_F1:
                if ez_mode:
                    if os.path.exists("model.npy"):
                        bird_list[0].q_values = np.load('model.npy')
                        ez_mode=False
                else:
                    if os.path.exists("ez.npy"):
                        bird_list[0].q_values = np.load('ez.npy')
                        ez_mode=True



    player.move()
    player.draw ()
    screen.blit(myFont.render(
                    (f'{player.score}'), True, (255, 255, 255)), (200, 60))
    if player.score > highscore:
        highscore = player.score
    if player.dead and game_over==False:
            game_over=True
            lose=True
    
    else:
        for bird in bird_list:
            if player.dead:
                break
            if not bird.dead:

                action_index = int(get_next_action(
                    bird.horizontal_dif, bird.height_dif, bird.lower, epsilon, bird.q_values))
                if action_index == 1:
                    bird.fly = True

                bird.move()
                bird.draw()



                
            else:

                if len(bird_list) > 1:
                    dead_birds.append(bird.q_values)
                    bird_list.remove(bird)
                elif (game_over==False):
                    win=True
                    game_over =True

    screen.blit(myFont.render(
        (f'Highscore: {highscore}'), True, (255, 255, 255)), (0, 0))
    if lose:
        screen.blit(myFont1.render(
            (f'Przegrałeś'), True, (255, 0, 0)), (0, 400))
        screen.blit(myFont.render(
            (f'Wciśnij spację'), True, (255, 0, 0)), (0, 500))
    if win:
        screen.blit(myFont1.render(
            (f'Wygrałeś'), True, (0, 0, 0)), (0, 400))
        screen.blit(myFont.render(
            (f'Wciśnij spację'), True, (0, 0, 0)), (0, 500))
    if ez_mode:
        pygame.draw.circle(screen,(255,0,0), (350,50),30)
    pygame.display.update()
    clock.tick(FPS)
