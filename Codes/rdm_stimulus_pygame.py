import pygame
import random
import numpy as np

SCREEN_SIZE = (800, 600)
DOTS_COUNT = 300
DOT_SIZE = 2
DRIVING_POWERS = [3.2, 6.4, 12.8, 25.6]
TRIAL_TIME_MS = 500  
BLOCKS_COUNT = 8
TRIALS_PER_BLOCK = 200

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()

def generate_dots():
    return [(random.randint(0, SCREEN_SIZE[0]), random.randint(0, SCREEN_SIZE[1])) for _ in range(DOTS_COUNT)]

def update_dots(dots, coherence_direction, driving_power):
    coherence_vector = np.array([np.cos(coherence_direction), np.sin(coherence_direction)])
    new_dots = []
    for dot in dots:
        if random.random() < (driving_power / 100):
            movement = coherence_vector * random.uniform(1, 3)
        else:
            movement = np.array([random.uniform(-3, 3), random.uniform(-3, 3)])
        
        new_dot = (dot[0] + movement[0], dot[1] + movement[1])
        
        new_dot = (new_dot[0] % SCREEN_SIZE[0], new_dot[1] % SCREEN_SIZE[1])
        new_dots.append(new_dot)
    return new_dots

def run_trial(driving_power):
    print("Running trial...")
    dots = generate_dots()
    coherence_direction = random.uniform(0, 2 * np.pi)
    start_time = pygame.time.get_ticks()
    
    while pygame.time.get_ticks() - start_time < TRIAL_TIME_MS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        screen.fill((0, 0, 0))
        
        dots = update_dots(dots, coherence_direction, driving_power)
        for dot in dots:
            pygame.draw.circle(screen, (255, 255, 255), (int(dot[0]), int(dot[1])), DOT_SIZE)
        
        pygame.display.flip()
        clock.tick(60)
    
    print("Trial completed.")
    return True

def run_block(driving_power, block_num):
    print(f"Running block {block_num + 1} with driving power {driving_power}...")
    for trial in range(TRIALS_PER_BLOCK):
        if not run_trial(driving_power):
            return False
        pygame.time.wait(200)  
        print(f"Completed trial {trial + 1} in block {block_num + 1}")
    print(f"Completed block {block_num + 1}.")
    return True

def main():
    for block in range(BLOCKS_COUNT):
        driving_power = random.choice(DRIVING_POWERS)
        print(f"Starting block {block + 1} with driving power {driving_power}")
        if not run_block(driving_power, block):
            break
        pygame.time.wait(1000) 
        print(f"Block {block + 1} completed.")

    pygame.quit()

if __name__ == "__main__":
    main()






