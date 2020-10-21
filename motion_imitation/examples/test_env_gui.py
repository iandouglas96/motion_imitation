"""Simple script for executing random actions on A1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error

from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import laikago
from motion_imitation.robots import robot_config
from scipy.spatial.transform import Rotation

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,
                    'Where to save video (or None for not saving).')

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}


def main(_):
  robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
  motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
  env = env_builder.build_regular_env(robot,
                                      motor_control_mode=motor_control_mode,
                                      enable_rendering=True,
                                      on_rack=FLAGS.on_rack,
                                      wrap_trajectory_generator=False)

  action_low, action_high = env.action_space.low/3., env.action_space.high/3.
  action_median = (action_low + action_high) / 2.
  dim_action = action_low.shape[0]
  action_selector_ids = []
  for dim in range(dim_action):
    action_selector_id = p.addUserDebugParameter(paramName='dim{}'.format(dim),
                                                 rangeMin=action_low[dim],
                                                 rangeMax=action_high[dim],
                                                 startValue=action_median[dim])
    action_selector_ids.append(action_selector_id)

  action_selector_id = p.addUserDebugParameter(paramName='height',
                                               rangeMin=0,
                                               rangeMax=2,
                                               startValue=1)
  action_selector_ids.append(action_selector_id)
  action_selector_id = p.addUserDebugParameter(paramName='angle',
                                               rangeMin=-3.14/2,
                                               rangeMax=3.14/2,
                                               startValue=0)
  action_selector_ids.append(action_selector_id)
  

  if FLAGS.video_dir:
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, FLAGS.video_dir)

  #for _ in tqdm(range(800)):
  last_state_vec = []
  while True:
    state_vec = []
    rot = Rotation.from_euler('zxy', [0,0,env.pybullet_client.readUserDebugParameter(
          action_selector_ids[-1])])
    base = Rotation.from_quat([0.5,0.5,0.5,0.5])
    final_quat = (rot*base).as_quat()
    final_pos = [0,0,env.pybullet_client.readUserDebugParameter(
          action_selector_ids[-2])]
    env.moveRack(final_pos, final_quat)

    state_vec.extend(final_pos)
    state_vec.extend(final_quat)

    action = np.zeros(dim_action)
    for dim in range(dim_action):
      action[dim] = env.pybullet_client.readUserDebugParameter(
          action_selector_ids[dim])
      state_vec.append(action[dim])
    env.step(action)

    if last_state_vec != state_vec:
        print(state_vec)
        last_state_vec = state_vec


  if FLAGS.video_dir:
    p.stopStateLogging(log_id)


if __name__ == "__main__":
  app.run(main)
