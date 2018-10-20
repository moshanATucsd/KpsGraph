import numpy as np
import matplotlib.pyplot as plt
import time
import glob

# np.random.seed(0)


class Synthetic(object):
    def __init__(self, n_kps=36, box_size=60., loc_std=.5):
        self.n_kps = n_kps
        self.box_size = box_size      
        self.loc_std = loc_std
        self._edge_types = np.array([-1., 0., 1.])
        self.Folder='/home/dinesh/Research/car_render/'
        
        



    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_keypoints(self, car_num=0, Folder = ''):
        self.Folder = Folder
        locs = list()
        viss = list()
        edges = list()
        paths = list()
        types = list()

        n = self.n_kps
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        keypoint_path = self.Folder +'/'+ str(car_num) + '/gt/'
        print(keypoint_path)
        files = glob.glob(keypoint_path + '*.txt')
        for loop,file_name in enumerate(files):
            kps_2d = np.loadtxt(file_name, delimiter=',')
            if n != 36:
                data_corr = [[0,34],[1,16],[2,35],[3,17],[4,19],[5,1],[6,24],[7,6],[10,22],[11,4],[8,21],[9,3]]
                kps_2d_new = np.zeros((n,4))
                for a,b in enumerate(data_corr):
                    kps_2d_new[b[0],0:2] = kps_2d[b[1],0:2]
                    kps_2d_new[b[0],2:3] = kps_2d[b[1],4:5]
                kps_2d = kps_2d_new
                vis_save = kps_2d[:,2:3].T
                vis = kps_2d[:,2].T
            else:
                vis_save = kps_2d[:,4:5].T
                vis = kps_2d[:,4].T
                
            
            loc = kps_2d[:,0:2].T
            #vis = kps_2d[:,2].T

            ty = np.zeros((self.n_kps,3))
            for i in range(n):
                ty[0,:] = [0,2,2] #back wheel
                ty[1,:] = [1,2,2] #back wheel
                ty[2,:] = [0,1,2] #back wheel
                ty[3,:] = [1,1,2] #back wheel
                
                ty[4,:] = [1,0,1] #back wheel
                ty[5,:] = [0,0,1] #back wheel
                ty[6,:] = [0,3,1] #back wheel
                ty[7,:] = [1,3,1] #back wheel
                
                ty[8,:] = [0,2,0] #back wheel
                ty[9,:] = [0,1,0] #back wheel
                ty[10,:] = [1,1,0] #back wheel
                ty[11,:] = [1,2,0] #back wheel
            

            edge = np.zeros((n,n))
            #print(keypoint_path,vis)
            for i in range(n):
                for j in range(n):
                    #if vis[i]==2 and vis[j]==2:
                    #    edge[i,j]=1
                    if vis[i]==1 or vis[j]==1:
                        edge[i,j]=0
                    else:
                        edge[i,j]=-1
            locs.append(loc)
            edges.append(edge)
            viss.append(vis_save)
            paths.append(file_name)
            types.append(ty)
                        
            #loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
        
        return locs, edges, viss, paths, types


class Carfusion(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_keypoints(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges


if __name__ == '__main__':
    sim = Synthetic()
    # sim = ChargedParticlesSim()

    t = time.time()
    loc, vel, edges, paths, types = sim.sample_keypoints()

    print(edges)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    # #plt.plot(vel_norm[:,i])
    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], edges) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    #     mom = vel.sum(axis=2)
    #     mom_diff = (mom[1:,:]-mom[:-1,:]).sum(axis=1)
    #     plt.figure()
    #     plt.plot(mom_diff)
    plt.show()

    # np.save("loc.npy", loc)
    # np.save("vel.npy", vel)
    # np.save("edges.npy", edges)
