import numpy as np
from lerobot.envs.pusht.pusht_image_env import PushTImageEnv
import pygame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE, REWARD

REPO_ID = "olegbalakhnov/pusht_images"
TASK = "PushT-v0"
NUM_EPISODES_TO_RECORD = 10
PUSH_TO_HUB = True
RESUME = True

render_size = 96
control_hz=10

# create PushT env with keypoints
env = PushTImageEnv(render_size=render_size)
agent = env.teleop_agent()
clock = pygame.time.Clock()

features = {
            OBS_STATE: {"dtype": "float32", "shape": (2,), "names": "observation.state"},
            OBS_IMAGE: {
                    "dtype": "video",
                    "shape": (96, 96, 3),
                    "names": ["channels", "height", "width"],
                },
            ACTION: {"dtype": "float32", "shape": (2,), "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            "next.success": {"dtype": "bool", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }
if RESUME:
    dataset = LeRobotDataset(
                REPO_ID
            )
else:
    dataset = LeRobotDataset.create(
                REPO_ID,
                env.metadata["video.frames_per_second"],
                use_videos=True,
                image_writer_threads=4,
                image_writer_processes=0,
                features=features,
            )

step = 0
episode_idx = 0
while episode_idx < NUM_EPISODES_TO_RECORD:
    # reset env and get observations (including info and render for recording)
    env.seed()
    prev_obs = env.reset()
    info = env._get_info()
    img = env.render(mode='human')
    
    # loop state
    retry = False
    pause = False
    done = False
    plan_idx = 0
    pygame.display.set_caption(f'plan_idx:{plan_idx}')
    # step-level while loop
    while not done:
        # process keypress events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # hold Space to pause
                    plan_idx += 1
                    pygame.display.set_caption(f'plan_idx:{plan_idx}')
                    pause = True
                elif event.key == pygame.K_r:
                    # press "R" to retry
                    retry=True
                elif event.key == pygame.K_q:
                    # press "Q" to exit
                    exit(0)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    pause = False
        # handle control flow
        if retry:
            break
        if pause:
            continue
        
        # get action from mouse
        # None if mouse is not close to the agent
        act = agent.act(prev_obs)
        # step env and render
        obs, reward, done, info = env.step(act)
        img = env.render(mode='human')
        if act is not None:
            prev_obs["image"] = np.moveaxis(prev_obs["image"], 0, -1)
            act = np.array(act)
            frame = {
                    OBS_IMAGE: prev_obs["image"],
                    OBS_STATE: obs["agent_pos"].astype(np.float32),
                    ACTION: act.astype(np.float32),
                    REWARD: np.array([reward], dtype=np.float32),
                    "next.success": np.array([done], dtype=bool), 
                    "next.done": np.array([done], dtype=bool), 
                    "task": TASK
                }
            
            dataset.add_frame(frame)
            
        prev_obs = obs
        # regulate control frequency
        clock.tick(control_hz)
    if not retry:
        episode_idx += 1
        dataset.save_episode()
        print(f'saved seed {env.seed}')
    else:
        print(f'retry seed {env.seed}')
        
if PUSH_TO_HUB:
    dataset.push_to_hub()