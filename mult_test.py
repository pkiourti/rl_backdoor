import os
from train import get_network_and_environment_creator, bool_arg
import subprocess
import argparse
import json
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
parser.add_argument('-tc', '--test_count', default='5', type=int, help="The amount of tests to run on the given network", dest="test_count")
parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
parser.add_argument('-gn', '--gif_name', default=None, type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name")
parser.add_argument('-gf', '--gif_folder', default='', type=str, help="The folder where to save gifs.", dest="gif_folder")

parser.add_argument('--checkpoints_foldername', default='poison_checkpoints', type=str, help='name of the checkpoints folder', dest='checkpoints_foldername')
parser.add_argument('--poison', default=False, type=bool_arg, help="Whether poison or not", dest="poison")
parser.add_argument('--action', default=1, type=int, help="specify the target action during training", dest="action")
parser.add_argument('--poison_once', default=False, type=bool_arg, help="Poison only once during testing", dest="poison_once")
parser.add_argument('--window', default=0, type=int, help="Poison after leaving window states without poisoning", dest="window")
parser.add_argument('--poison_every_other', default=False, type=bool_arg, help="Poison every other state", dest="poison_every_other")
parser.add_argument('--pixels_to_poison', default=3, type=int, help="pixels that will be poisoned", dest="pixels_to_poison")
parser.add_argument('--model_step', default=100, type=int, help="step that will be used to go to the next model", dest="step")

args = parser.parse_args()

folder = os.path.join(args.folder, args.checkpoints_foldername)
ls = os.listdir(folder)
numbers = []

for f in ls:
    if "meta" in f:
        numbers.append(f[11:-5])

numbers.sort(key=int)

for poison in [True, False]:
    rewards = []
    #times = []
    for model in tqdm(numbers[::args.step]):
        argslist = ['python3', 'test.py', '-f', args.folder, 
                '--checkpoints_foldername', args.checkpoints_foldername, 
                '--poison', poison, '--pixels_to_poison', args.pixels_to_poison, 
                '--action', args.action, '--poison_once', args.poison_once,
                '--index', model, '-tc', args.test_count, '--window', args.window,
                '--poison_every_other', args.poison_every_other]
        argslist = [str(s) for s in argslist]
        lines = subprocess.check_output(argslist, stderr=subprocess.DEVNULL).decode('utf-8').split('\n')
        # stderr=subprocess.DEVNULL
        for l in lines:
            if "Mean" in l:
                rewards.append((int(model), float(l[6:])))
            #if "elapsed" in l:
            #    print((int(model), float(l[14:])))
            #    times.append((int(model), float(l[14:])))

    with open(os.path.join(args.folder, 'results_clean.json') if not poison else os.path.join(args.folder, 'results_poison.json'), 'w') as f:
        json.dump(rewards, f)

    #with open(os.path.join(args.folder, 'time_clean.json') if not poison else os.path.join(args.folder, 'time_poison.json'), 'w') as f:
    #    json.dump(times, f)
