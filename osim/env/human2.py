import opensim as osim
import math
import numpy as np
from gym import spaces
import os
from environments.osim import OsimEnv

import random

class GaitEnv(OsimEnv):
    ninput = 24
    model_path = os.path.join(os.path.dirname(__file__), '../../models/gait9dof18musc_Thelen_3Spheres_20161108.osim') #'../../models/gait3spheres.osim'

    def compute_reward_standing(self):
        y = self.ground_pelvis.getCoordinate(2).getValue(self.state)
        x = self.ground_pelvis.getCoordinate(1).getValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        acc = self.model.calcMassCenterAcceleration(self.state)

        rew = 100 - abs(acc[0])**2 - abs(acc[1])**2 - abs(acc[2])**2 - abs(vel[0])**2 - abs(vel[1])**2 - abs(vel[2])*2*t

        obs = self.get_observation()
        ext = 100 * sum([x**2 for x in obs]) / self.noutput
        rew = rew - ext
        
        if rew < -100:
            rew = -100
        return rew / 100.0

    def compute_reward(self):
        obs = self.get_observation()
	dist = self.joints[0].getCoordinate(1).getValue(self.state)
		
	rew = 0.1 * dist * dist
	if dist < 0:
		rew = -rew
        return rew

    def is_head_too_low(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y < 0.45)
    
    def is_done(self):
        return self.is_head_too_low()

    def __init__(self, visualize = True):
        super(GaitEnv, self).__init__(visualize = visualize)

        self.joints.append(osim.PlanarJoint.safeDownCast(self.jointSet.get(0))) # PELVIS

        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(1)))
        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(2))) # 4
        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(3)))    # 7
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(4)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(5)))

        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(6)))    # 2
        self.joints.append(osim.CustomJoint.safeDownCast(self.jointSet.get(7))) # 5
        self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(8)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(9)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(10)))

        # self.joints.append(osim.PinJoint.safeDownCast(self.jointSet.get(11)))
        # self.joints.append(osim.WeldJoint.safeDownCast(self.jointSet.get(12)))

        self.head = self.bodySet.get(12)

        # pos = self.model.calcMassCenterPosition(self.state)
        # self.hy = pos[1]

        self.reset()

    def reset(self):
        self.istep = 0
        if not self.state0:
            self.state0 = self.model.initSystem()
            self.manager = osim.Manager(self.model)
            self.state = osim.State(self.state0)
        else:
            self.state = osim.State(self.state0)

        self.model.equilibrateMuscles(self.state)
        self.prev_reward = 0
	
#	velRange = -0.2
#       	vel = random.uniform(-velRange, velRange)
	vel = -0.5
	print "\n Random start velocity = %f" % vel

#	print "\n Joints2 size = %i" % len(self.joints)
	if len(self.joints) > 0:
		self.joints[0].getCoordinate(1).setSpeedValue(self.state, vel)
		
        pos = self.model.calcMassCenterPosition(self.state)
        self.hy = pos[1]
        # print "\n hy = %f" % self.hy

        # nullacttion = np.array([0] * self.noutput, dtype='f')
        # for i in range(0, int(math.floor(0.2 / self.stepsize) + 1)):
        #     self.step(nullacttion)
 
        return [0.0] * self.ninput # self.get_observation()

    def get_observation(self):
        invars = np.array([0] * self.ninput, dtype='f')

        invars[0] = self.joints[0].getCoordinate(0).getValue(self.state)
        invars[1] = self.joints[0].getCoordinate(1).getValue(self.state)
        invars[2] = self.joints[0].getCoordinate(2).getValue(self.state)

        invars[3] = self.joints[0].getCoordinate(0).getSpeedValue(self.state)
        invars[4] = self.joints[0].getCoordinate(1).getSpeedValue(self.state)
        invars[5] = self.joints[0].getCoordinate(2).getSpeedValue(self.state)

        for i in range(6):
            invars[6+i] = self.joints[1+i].getCoordinate(0).getValue(self.state)
        for i in range(6):
            invars[12+i] = self.joints[1+i].getCoordinate(0).getSpeedValue(self.state)

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        
        invars[18] = pos[0]
        invars[19] = pos[1]
        invars[20] = pos[2]

        invars[21] = vel[0]
        invars[22] = vel[1]
        invars[23] = vel[2]

        for i in range(0, self.ninput):
            invars[i] = self.sanitify(invars[i])

        return invars

class StandEnv(GaitEnv):
    def compute_reward(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        x = self.joints[0].getCoordinate(1).getValue(self.state)
        alpha = self.joints[0].getCoordinate(0).getValue(self.state)
        #print "\n alpha = %f" % alpha

        pos = self.model.calcMassCenterPosition(self.state)
        vel = self.model.calcMassCenterVelocity(self.state)
        acc = self.model.calcMassCenterAcceleration(self.state)

        rew = 100 - abs(acc[0])**2 - abs(acc[1])**2 - abs(acc[2])**2 - abs(vel[0])**2 - abs(vel[1])**2 - abs(vel[2])**2 - 50.0 * abs(alpha)

        obs = self.get_observation()
        ext = 100 * sum([x**2 for x in obs]) / self.noutput
        rew = rew - ext

        y_cm = pos[1]
        dy = self.hy - y_cm
        ry = 2.0 * dy

        if dy < 0.0:
            ry = -5.0 * dy

    #    rew = rew - ry
        
        if rew < -100:
            rew = -100
        return rew / 100.0

class HopEnv(GaitEnv):
    def compute_reward(self):
        pos = self.model.calcMassCenterPosition(self.state)
        y = pos[1]
        ry = self.hy - y

        if ry > 0.0:
            return sqrt(y + 0.5) / 100.0
        else:
            return (y + 0.5)**10 / 100.0
  #      y = self.joints[0].getCoordinate(2).getValue(self.state)
  #      return y**2

    def is_head_too_low(self):
        y = self.joints[0].getCoordinate(2).getValue(self.state)
        return (y < 0.3)
