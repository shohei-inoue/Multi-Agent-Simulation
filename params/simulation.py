from params.environment import EnvironmentParam
from params.robot import RobotParam
from params.explore import ExploreParam
from params.agent import AgentParam

class Param:
  environment : EnvironmentParam = EnvironmentParam()
  agent       : AgentParam       = AgentParam()
  robot       : RobotParam       = RobotParam()
  explore     : ExploreParam     = ExploreParam()