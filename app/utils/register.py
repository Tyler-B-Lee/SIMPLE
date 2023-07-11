

def get_environment(env_name):
    try:
        if env_name in ('tictactoe'):
            from tictactoe.envs.tictactoe import TicTacToeEnv
            return TicTacToeEnv
        elif env_name in ('connect4'):
            from connect4.envs.connect4 import Connect4Env
            return Connect4Env
        elif env_name in ('ttykm'):
            from ttykm.envs.ttykm import ttykmEnv
            return ttykmEnv
        elif env_name in ('ttykmch1'):
            from ttykmch1.envs.ttykmch1 import ttykmch1Env
            return ttykmch1Env
        elif env_name in ('onitama'):
            from onitama.envs.onitama import OnitamaEnv
            return OnitamaEnv
        elif env_name in ('shobu'):
            from shobu.envs.shobu import ShobuEnv
            return ShobuEnv
        elif env_name in ('quarto'):
            from quarto.envs.quarto import QuartoEnv
            return QuartoEnv
        elif env_name in ('root2pCatsVsEyrie'):
            from root2pCatsVsEyrie.envs.root2pCatsVsEyrie import rootEnv
            return rootEnv
        elif env_name in ('root2pCVEWinter'):
            from root2pCVEWinter.envs.root2pCVEWinter import rootEnv
            return rootEnv
        elif env_name in ('sushigo'):
            from sushigo.envs.sushigo import SushiGoEnv
            return SushiGoEnv
        elif env_name in ('sushino'):
            from sushino.envs.sushino import SushiNOEnv
            return SushiNOEnv
        elif env_name in ('butterfly'):
            from butterfly.envs.butterfly import ButterflyEnv
            return ButterflyEnv
        elif env_name in ('geschenkt'):
            from geschenkt.envs.geschenkt import GeschenktEnv
            return GeschenktEnv
        elif env_name in ('frouge'):
            from frouge.envs.frouge import FlammeRougeEnv
            return FlammeRougeEnv
        elif env_name in ('elements'):
            from elements.envs.elements import ElementsEnv
            return ElementsEnv
        elif env_name in ('mancala'):
            from mancala.envs.mancala import MancalaEnv
            return MancalaEnv
        elif env_name in ('antimancala'):
            from antimancala.envs.antimancala import AntiMancalaEnv
            return AntiMancalaEnv
        elif env_name in ('minecraftcg'):
            from minecraftcg.envs.minecraftcg import MinecraftCGEnv
            return MinecraftCGEnv
        elif env_name in ('brandubh'):
            from brandubh.envs.brandubh import BrandubhEnv
            return BrandubhEnv
        else:
            raise Exception(f'No environment found for {env_name}')
    except SyntaxError as e:
        print(e)
        raise Exception(f'Syntax Error for {env_name}!')
    except:
        raise Exception(f'Install the environment first using: \nbash scripts/install_env.sh {env_name}\nAlso ensure the environment is added to /utils/register.py')
    

def get_network_arch(env_name):
    if env_name in ('tictactoe'):
        from models.tictactoe.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('connect4'):
        from models.connect4.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('ttykm'):
        from models.ttykm.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('ttykmch1'):
        from models.ttykmch1.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('onitama'):
        from models.onitama.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('shobu'):
        from models.shobu.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('quarto'):
        from models.quarto.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('root2pCatsVsEyrie'):
        from models.root2pCatsVsEyrie.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('root2pCVEWinter'):
        from models.root2pCVEWinter.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('sushigo'):
        from models.sushigo.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('sushino'):
        from models.sushino.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('butterfly'):
        from models.butterfly.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('geschenkt'):
        from models.geschenkt.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('frouge'):
        from models.frouge.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('elements'):
        from models.elements.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('mancala'):
        from models.mancala.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('antimancala'):
        from models.antimancala.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('minecraftcg'):
        from models.minecraftcg.models import CustomPolicy
        return CustomPolicy
    elif env_name in ('brandubh'):
        from models.brandubh.models import CustomPolicy
        return CustomPolicy
    else:
        raise Exception(f'No model architectures found for {env_name}')

