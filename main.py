import pygame
import random
import numpy as np

pygame.init()


class NeuralNetwork:
    def __init__(self, x, y, z, weights1, weights2, bias1, bias2):
        self.x = x
        self.y = y
        self.z = z
        self.weights1 = weights1
        self.weights2 = weights2
        self.bias1 = bias1
        self.bias2 = bias2

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def predict(self):
        data = np.array([[self.x], [self.y], [self.z]])
        layer2 = self.sigmoid(np.dot(self.weights1, data) + self.bias1)
        layer3 = self.sigmoid(np.dot(self.weights2, layer2) + self.bias2)
        return layer3


class Bird:
    def __init__(self, bird_image, bird_mask, weights1=np.random.rand(7, 3) - 0.5,
                 weights2=np.random.rand(1, 7) - 0.5, bias1=np.random.rand(7, 1) - 0.5,
                 bias2=np.random.rand(1, 1) - 0.5):
        self.x = 50
        self.y = 50
        self.gravity = 0.75
        self.acceleration = 0.075
        self.image = bird_image
        self.mask = bird_mask
        self.weight1 = weights1
        self.weight2 = weights2
        self.bias1 = bias1
        self.bias2 = bias2

        self.score = 0
        self.pipe_y = 0
        self.pipe_distance = 0
        self.distance_to_pipe = 0
        self.pipe_height = 0

        self.network = NeuralNetwork(
            self.y, self.distance_to_pipe, self.pipe_height,
            self.weight1, self.weight2, self.bias1, self.bias2
        )

    def draw(self, window):
        window.blit(self.image, (int(self.x), int(self.y)))

    def update_network(self):
        self.network.x = self.y
        self.network.y = self.distance_to_pipe
        self.network.z = self.pipe_height

    def update(self):
        if self.y < 0:
            return "DEAD"
        elif self.y < 380:
            self.y += self.gravity
            self.gravity += self.acceleration
        else:
            return "DEAD"
        return "ALIVE"

    def jump(self):
        if self.network.predict() < 0.5:
            self.gravity = -2


class Pipe:
    def __init__(self, x, y, pipe_id, image):
        self.x = x
        self.lower_y = y
        self.upper_y = self.lower_y - 420
        self.id = pipe_id
        self.lower_image = image
        self.upper_image = pygame.transform.flip(self.lower_image, False, True)

        self.lower_mask = pygame.mask.from_surface(self.lower_image)
        self.upper_mask = pygame.mask.from_surface(self.upper_image)

    def draw(self, window):
        window.blit(self.lower_image, (self.x, self.lower_y))
        window.blit(self.upper_image, (self.x, self.upper_y))

    def move(self, game_speed):
        self.x -= game_speed


class GameCore:
    def __init__(self, population_size=1):
        self.window_height = 288
        self.window_width = 512
        self.window = pygame.display.set_mode((self.window_height, self.window_width))
        self.clock = pygame.time.Clock()
        self.game_speed = 2

        # Load assets
        self.background = pygame.image.load("assets/background.png").convert()
        self.base = pygame.image.load("assets/base.png").convert()
        self.pipe_image = pygame.image.load("assets/pipe.png").convert_alpha()
        self.bird_image = pygame.image.load("assets/bird.png").convert_alpha()
        self.bird_mask = pygame.mask.from_surface(self.bird_image)

        self.font = pygame.font.SysFont("Arial", 40)
        self.pipe_id = 4

        # Initialize pipes
        self.pipes = self._create_initial_pipes()

        # Genetic algorithm variables
        self.population = []
        self.next_generation = []
        self.population_size = population_size
        self.dead_birds = []
        self.generation = 0

        self._initialize_population()

    def _create_initial_pipes(self):
        return [
            Pipe(300, random.randint(220, 340), 0, self.pipe_image),
            Pipe(460, random.randint(220, 340), 1, self.pipe_image),
            Pipe(620, random.randint(220, 340), 2, self.pipe_image),
            Pipe(780, random.randint(220, 340), 3, self.pipe_image)
        ]

    def _initialize_population(self):
        for _ in range(self.population_size):
            weights = self._create_weights()
            self.population.append(Bird(self.bird_image, self.bird_mask, *weights))

    def check_collision(self, masked_obj1, pos1_x, pos1_y, masked_obj2, pos2_x, pos2_y):
        offset = (round(pos2_x - pos1_x), round(pos2_y - pos1_y))
        return masked_obj1.overlap(masked_obj2, offset)

    def draw(self):
        self.window.blit(self.background, (0, 0))
        self.window.blit(self.base, (0, 400))

        for pipe in self.pipes:
            pipe.draw(self.window)

        for bird in self.population:
            bird.draw(self.window)

        generation_text = self.font.render(str(self.generation), True, (255, 255, 255))
        self.window.blit(generation_text, (20, 20))

        self.clock.tick(60)
        pygame.display.update()

    def _create_weights(self):
        weights1 = np.random.rand(7, 3) - 0.5
        weights2 = np.random.rand(1, 7) - 0.5
        bias1 = np.random.rand(7, 1) - 0.5
        bias2 = np.random.rand(1, 1) - 0.5
        return weights1, weights2, bias1, bias2

    def crossover(self):
        self.dead_birds = sorted(self.dead_birds, key=lambda bird: bird.score)

        if self.dead_birds[-1].score == 0:
            self.create_new_generation()
        else:
            self.next_generation = []
            last_best = int((98 * self.population_size) / 100)
            self.next_generation = []
            self.next_generation.extend(self.population[last_best:])
            for member in self.next_generation:
                member.x = 50
                member.y = 50
                member.score = 0

            while True:
                if len(self.next_generation) < self.population_size:
                    member_1 = random.choice(self.dead_birds[last_best:])
                    member_1_weight_1 = member_1.weight1
                    member_1_weight_2 = member_1.weight2
                    member_1_bias_1 = member_1.bias1
                    member_1_bias_2 = member_1.bias2

                    member_2 = random.choice(self.dead_birds[last_best:])
                    member_2_weight_1 = member_2.weight1
                    member_2_weight_2 = member_2.weight2
                    member_2_bias_1 = member_2.bias1
                    member_2_bias_2 = member_2.bias2

                    chield_1_weight_1 = []
                    chield_1_weight_2 = []
                    chield_1_bias_1 = []
                    chield_1_bias_2 = []

                    for x, y in zip(member_1_weight_1, member_2_weight_1):
                        for i, k in zip(x, y):
                            Prob = random.random()
                            if Prob < 0.47:
                                chield_1_weight_1.append(i)
                            elif Prob < 0.94:
                                chield_1_weight_1.append(k)
                            else:
                                chield_1_weight_1.append(random.uniform(-0.5, 0.5))

                    for a, b in zip(member_1_weight_2, member_2_weight_2):
                        for c, d in zip(a, b):
                            Prob = random.random()
                            if Prob < 0.47:
                                chield_1_weight_2.append(c)
                            elif Prob < 0.94:
                                chield_1_weight_2.append(d)
                            else:
                                chield_1_weight_2.append(random.uniform(-0.5, 0.5))

                    for t, y in zip(member_1_bias_1, member_2_bias_1):
                        for v, b in zip(t, y):
                            Prob = random.random()
                            if Prob < 0.47:
                                chield_1_bias_1.append(v)
                            elif Prob < 0.94:
                                chield_1_bias_1.append(b)
                            else:
                                chield_1_bias_1.append(random.uniform(-0.5, 0.5))

                    for q, w in zip(member_1_bias_2, member_2_bias_2):
                        Prob = random.random()
                        if Prob < 0.47:
                            chield_1_bias_2.append(q)
                        elif Prob < 0.94:
                            chield_1_bias_2.append(w)
                        else:
                            chield_1_bias_2.append(random.uniform(-0.5, 0.5))

                    chield_1_weight_1 = np.array(chield_1_weight_1)
                    chield_1_weight_2 = np.array(chield_1_weight_2)

                    chield_1_bias_1 = np.array(chield_1_bias_1)
                    chield_1_bias_2 = np.array(chield_1_bias_2)

                    chield_1_weight_1 = chield_1_weight_1.reshape((7, 3))
                    chield_1_weight_2 = chield_1_weight_2.reshape((1, 7))
                    chield_1_bias_1 = chield_1_bias_1.reshape((7, 1))
                    chield_1_bias_2 = chield_1_bias_2.reshape((1, 1))

                    self.next_generation.append(Bird(self.bird_image, self.bird_mask,
                                                     chield_1_weight_1, chield_1_weight_2,
                                                     chield_1_bias_1, chield_1_bias_2))

                else:
                    break

            self.population = self.next_generation

    def create_new_generation(self):
        for _ in range(self.population_size):
            weights = self._create_weights()
            self.population.append(Bird(self.bird_image, self.bird_mask, *weights))

    def restart_game(self):
        self.pipes = self._create_initial_pipes()
        self.crossover()

    def game_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "Close"

        self.Tus = pygame.key.get_pressed()
        if self.Tus[pygame.K_ESCAPE]:
            return "Close"

        self.FPS = str(int(self.clock.get_fps()))
        pygame.display.set_caption(f"Fps : {self.FPS}")

        if len(self.population) != 0:
            print(f"Best Score = {self.pipe_id}, prediction = {self.population[-1].network.predict()}, alive = {len(self.population)}")

        for pipe in self.pipes:
            if pipe.x == -52:
                pipe.x = 588
                pipe.lower_y = random.randint(220, 340)
                pipe.upper_y = pipe.lower_y - 420
                pipe.id = self.pipe_id
                self.pipe_id += 1

            pipe.move(self.game_speed)

            for bird in self.population:
                if bird.score == pipe.id:
                    bird.distance_to_pipe = pipe.x - bird.x
                    bird.pipe_height = pipe.lower_y
                    bird.update_network()

                collision_lower = self.check_collision(bird.mask, bird.x, bird.y,
                                                        pipe.lower_mask, pipe.x, pipe.lower_y)
                collision_upper = self.check_collision(bird.mask, bird.x, bird.y,
                                                        pipe.upper_mask, pipe.x, pipe.upper_y)

                if collision_lower != None or collision_upper != None:
                    self.dead_birds.append(bird)
                    self.population.remove(bird)

                if bird.x + 16 == pipe.x:
                    bird.score += 1

        for bird in self.population:
            if bird.update() == "DEAD":
                self.dead_birds.append(bird)
                self.population.remove(bird)

            bird.jump()

        if len(self.population) == 0:
            self.generation += 1
            self.restart_game()
            self.pipe_id = 4

        self.draw()


Population_Number = 300


Game = GameCore(Population_Number)
while True:
    GameStatus = Game.game_loop()
    if GameStatus == "Close":
        break

pygame.quit()