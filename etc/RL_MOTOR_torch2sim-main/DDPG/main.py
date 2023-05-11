import argparse
from datetime import datetime
from simManager import SimManager
from agent_train import train



def main(args):

    print("[MAIN START]")

    modelName = args.slx_name

    env = SimManager(modelName)
    env.connectMatlab()

    if args.train:
        train(args, env)
    else:
        print("No constructed NOT LOADED")

    env.disconnectMatlab()






if __name__ == "__main__":
    now = datetime.now()
    now = now.strftime('%m%d%H%M')

    parser = argparse.ArgumentParser(description = "Available Options")
    parser.add_argument('--train_name', type=str,
                        default=now, dest="train_name", action="store",
                        help='trained model would be saved in typed directory')
    parser.add_argument('--slx_name', type=str,
                        default='RL_Traction_motor_torch', dest='slx_name', action="store",
                        help='simulink model that you want to use as envirnoment')
    parser.add_argument('--train',
                        default=True, dest='train', action='store_true',
                        help='type this arg to train')
    parser.add_argument('--load_network', type=str,
                        default='NOT_LOADED', dest='load_network_path', action='store',
                        help='to load trained network, add path')
    parser.add_argument('--parameters', type=str,
                        default='./train_param.json', dest='param_path', action="store",
                        help='type path of train parameters file (json)')
    args = parser.parse_args()

    main(args)
