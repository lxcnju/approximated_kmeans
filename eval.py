# -*- coding : utf-8 -*-

### some methods to evaluate kmeans clustering results

from matplotlib import pyplot as plt
import numpy as np


class ClusterEval():
    def __init__(self, data, clu_labels, labels = None):
        ### init function
        ### @params data : numpy.array source data
        ### @params clu_labels : numpy.array cluster labels
        ### @params labels : source labels, if no labels available, None
        self.data = data
        self.clu_labels = clu_labels
        self.labels = labels

        self.n_data = data.shape[0]                   # data number
        self.n_clusters = len(np.unique(clu_labels))  # the number of clusters
        if isinstance(labels, np.ndarray):
            self.n_classes = len(np.unique(labels))   # real class number

        self.centers = self.calc_centers()            # find centers

        self.CP = self.compactness()
        self.SP = self.separation()

        self.labels_grid = self.calc_labels_grid()   # labels fusion matrix
        self.norm_labels_grid = self.labels_grid / np.sum(self.labels_grid, axis = 1).reshape(-1, 1)    # normlize

        self.RI = self.rand_index()
        self.ARI = self.adjust_rand_index()

    def calc_centers(self):
        ### calculate centers using clu_labels and data
        ### @return numpy.array
        centers = []
        for k in range(self.n_clusters):
            centers.append(np.mean(self.data[self.clu_labels == k, :]))
        centers = np.array(centers)
        print(centers.shape)
        return centers

    def compactness(self):
        ### compute the target function, eval inner-cluster distance
        ### the lower the better
        CP = 0.0
        for k in range(self.n_clusters):
            indexes = np.array(range(self.n_data))[self.clu_labels == k]
            clu_data = self.data[indexes, :]
            center = self.centers[k]
            CP += np.mean(np.sum(np.square(clu_data - center.reshape(1, -1)), axis = 1))
        return CP

    def separation(self):
        ### compute the between-cluster distance
        ### the higher the better
        SP = 0.0
        for k in range(self.n_clusters):
            dis2 = np.sum(np.square(self.centers - self.centers[k].reshape(1, -1)), axis = 1)
            SP += np.sum(np.sqrt(dis2))
        SP = 2 * SP / (self.n_clusters * (self.n_clusters - 1))
        return SP

    def calc_labels_grid(self):
        ### labels available, compute labels fusion matrix
        ### row-axis is cluster labels, col-axis is real labels
        if not isinstance(self.labels, np.ndarray):
            return None
        grid = np.zeros((self.n_clusters, self.n_clusters))
        for k in range(self.n_clusters):
            indexes = np.array(range(self.n_data))[self.clu_labels == k]
            real_labels = self.labels[indexes]
            for j in range(self.n_classes):
                grid[k][j] = np.sum(real_labels == j)
        return grid


    def rand_index(self):
        ### labels available, rand index
        ### the higher the better
        if not isinstance(self.labels, np.ndarray):
            return None
        # brute force, for every pair
        #tp = 0    # true positive, same cluster clustered in the same cluster
        #tn = 0    # true negative, different cluster clustered in the different cluster
        #for i in range(self.n_data):
        #    for j in range(self.n_data):
        #        if self.labels[i] == self.labels[j] and self.clu_labels[i] == self.clu_labels[j]:
        #            tp += 1
        #        if self.labels[i] != self.labels[j] and self.clu_labels[i] != self.clu_labels[j]:
        #            tn += 1
        #RI = 2.0 * (tp + tn)/(self.n_data * (self.n_data - 1))

        RI = 0.0
        for i in range(self.n_clusters):
            for j in range(self.n_classes):
                a = self.labels_grid[i][j]
                RI += a * (a - 1) / 2
        RI = RI / (self.n_data * (self.n_data - 1))
        return RI

    def adjust_rand_index(self):
        ### labels available, adjust rand index
        ### ARI = (RI - E[RI]) / (MaxRI -E[RI])
        ### the higher the better
        if not isinstance(self.labels, np.ndarray):
            return None
        sum_labels = np.sum(self.labels_grid, axis = 0)       # sum by col
        sum_clu_labels = np.sum(self.labels_grid, axis = 1)   # sum by row
        
        Index = 0                  # RI
        ExpectedIndex = 0          # E[RI]
        MaxIndex = 0               # MaxRI
        
        # calculate RI
        for i in range(self.n_clusters):
            for j in range(self.n_classes):
                a = self.labels_grid[i][j]
                Index += a * (a - 1)/2
        
        # calculate E[RI] and MaxRI
        sum_a = sum([x * (x - 1) / 2 for x in sum_labels])
        sum_b = sum([x * (x - 1) / 2 for x in sum_clu_labels])
        ExpectedIndex = 2 * sum_a * sum_b / (self.n_data * (self.n_data - 1))
        MaxIndex = (sum_a + sum_b) / 2

        ARI = (Index - ExpectedIndex) / (MaxIndex - ExpectedIndex)
        return ARI

