from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Door, Key, Lava
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

class NineRoomsEnv(MiniGridEnv):

    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ## Mission Space

    "reach the goal"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-FourRooms-v0`

    """

    def __init__(self, agent_pos=None, goal_pos=None, max_steps=500, task = 'keyeasy', **kwargs):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos

        self.size = 10
        self.task = task
        print("task is: ", self.task)
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "ninerooms"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 3
        room_h = height // 3
        self.grid.horz_wall(2,3,7)
        self.grid.horz_wall(3,2,1)
        self.grid.horz_wall(2,5,7, Lava)
        doorpos = (3,1)

        if self.task =='easykey' or self.task == 'hardkey' or self.task == 'keygoal':
            self._agent_default_pos = (1, 7)
        elif self.task == 'easykeygoal' or self.task == 'easykeydoor' :
            self._agent_default_pos = (7,7)
        elif self.task == 'hardkeygoal' or self.task == 'hardkeydoor':
            self._agent_default_pos = (1, 1)
        elif self.task == 'doorgoal':
            self._agent_default_pos = (4, 1)

        self.agent_pos = self._agent_default_pos
        self.grid.set(*self._agent_default_pos, None)
        self.agent_dir = self._rand_int(0, 4)
        # # Randomize the player start position and orientation
        # if self._agent_default_pos is not None:
        #     # assuming random start direction
        # else:
        #     self.place_agent()

        # self.grid.vert_wall(6, 9, 3)


        # goal_x = np.random.randint(low = self.size - room_w, high = self.size-2)
        # goal_y = np.random.randint(low = 1, high = room_h-1)
        self._goal_default_pos = (7,1)
        goal = Goal()
        self.put_obj(goal, *self._goal_default_pos)
        goal.init_pos, goal.cur_pos = self._goal_default_pos
        # if self._goal_default_pos is not None:
        # else:
        #     self.place_obj(Goal())

        # # Place a door in the wall
        # doorIdx = self._rand_int(1, width - 2)
        self.doorLocked = Door("yellow", is_locked=True)
        self.doorUnlocked = Door("yellow", is_open=True) 
        if self.task == 'doorgoal':
            self.put_obj(self.doorUnlocked, doorpos[0], doorpos[1])
        else:
            self.put_obj(self.doorLocked, doorpos[0], doorpos[1])

        # Place a yellow key on the left side
        self.keyHard = Key("yellow", 'keyHard')
        self.keyEasy = Key("yellow", 'keyEasy')
        if self.task =='easykey' or self.task == 'hardkey' or self.task == 'keygoal':
            self.place_obj(obj=self.keyHard, top=(1, 1), size=(1,1)) 
            self.place_obj(obj=self.keyEasy, top=(7,7), size=(1, 1))        

            # self.place_obj(obj=self.keyEasy, top=(self.size-2*room_h-1, self.size-room_w-1), size=(2, 2))        
            # self.place_obj(obj=self.keyEasy, top=(4,9), size=(2,2))        

        elif self.task == 'easykeygoal' or self.task == 'easykeydoor':
            self.place_obj(obj=self.keyHard, top=(0, 0), size=(room_h, room_w-1))        
        elif self.task == 'hardkeygoal' or self.task == 'hardkeydoor':
            self.place_obj(obj=self.keyEasy, top=(7,7), size=(1, 1))        
