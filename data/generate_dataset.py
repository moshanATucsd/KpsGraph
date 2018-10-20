from synthetic_sim import Synthetic, Carfusion
import time
import numpy as np
import argparse
from torch.distributions import normal
parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='single12',
                    help='What simulation to generate.')
parser.add_argument('--num-train', type=int, default=300,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=72,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=100,
                    help='Number of test simulations to generate.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--Folder', type=str, default='/home/dinesh/Research/car_render/',
                    help='Folder Path')


args = parser.parse_args()

if args.simulation == 'synthetic':
    sim = Synthetic()
    suffix = '_synthetic'
    train_start=0
    test_start=300
    valid_start=400
elif args.simulation == 'carfusion':
    sim = Carfusion()
    suffix = '_carfusion'
elif args.simulation == 'synthetic12':
    sim = Synthetic()
    sim.n_kps=12
    suffix = '_synthetic12'
    train_start=0
    test_start=300
    valid_start=400
elif args.simulation == 'single12':
    sim = Synthetic()
    sim.n_kps=12
    suffix = '_single12'
    train_start=1
    test_start=0
    valid_start=2
    args.num_train=1
    args.num_test=1
    args.num_valid=1
    args.Folder='/home/dinesh/Research/renders/'
else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

#suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims,start):
    loc_all = list()
    edges_all = list()
    viss_all = list()
    paths_all = list()
    types_all = list()

    for i in range(num_sims):
        t = time.time()
        i = i+start
        locs, edges,viss,paths,types = sim.sample_keypoints(i , Folder= args.Folder)
        if i % 1 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        for loc in enumerate(locs):
            loc_all.append(loc[1])
        for edge in enumerate(edges):
            edges_all.append(edge[1])
        for vis in enumerate(viss):
            viss_all.append(vis[1])
        for path in enumerate(paths):
            paths_all.append(path[1])
        for ty in enumerate(types):
            types_all.append(ty[1])

    loc_all = np.stack(loc_all)
    edges_all = np.stack(edges_all)
    viss_all = np.stack(viss_all)
    paths_all =np.stack(paths_all)
    types_all =np.stack(types_all)
    #asas

    return loc_all, edges_all, viss_all, paths_all, types_all


print("Generating {} training simulations".format(args.num_train))
loc_train, edges_train, vis_train, paths_train, types_train = generate_dataset(args.num_train, train_start)

print("Generating {} test simulations".format(args.num_test))
loc_test, edges_test, vis_test, paths_test, types_test = generate_dataset(args.num_test, test_start)

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, edges_valid, vis_valid, paths_valid, types_valid  = generate_dataset(args.num_valid, valid_start)

np.save('loc_train' + suffix + '.npy', loc_train)
np.save('edges_train' + suffix + '.npy', edges_train)
np.save('vis_train' + suffix + '.npy', vis_train)
np.save('path_train' + suffix + '.npy', paths_train)
np.save('type_train' + suffix + '.npy', types_train)

np.save('loc_valid' + suffix + '.npy', loc_valid)
np.save('edges_valid' + suffix + '.npy', edges_valid)
np.save('vis_valid' + suffix + '.npy', vis_valid)
np.save('path_valid' + suffix + '.npy', paths_valid)
np.save('type_valid' + suffix + '.npy', types_valid)

np.save('loc_test' + suffix + '.npy', loc_test)
np.save('edges_test' + suffix + '.npy', edges_test)
np.save('vis_test' + suffix + '.npy', vis_test)
np.save('path_test' + suffix + '.npy', paths_test)
np.save('type_test' + suffix + '.npy', types_test)
