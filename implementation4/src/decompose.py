import numpy as np
import copy


class PCA():
    """
    PCA. A class to reduce dimensions
    """

    def __init__(self, retain_ratio):
        """

        :param retain_ratio: percentage of the variance we maitain (see slide for definition)
        """
        self.retain_ratio = retain_ratio

    @staticmethod
    def mean(x):
        """
        returns mean of x
        :param x: matrix of shape (n, m)
        :return: mean of x of with shape (m,)
        """
        return x.mean(axis=0)

    @staticmethod
    def cov(x):
        """
        returns the covariance of x,
        :param x: input data of dim (n, m)
        :return: the covariance matrix of (m, m)
        """
        return np.cov(x.T)

    @staticmethod
    def eig(c):
        """
        returns the eigval and eigvec
        :param c: input matrix of dim (m, m)
        :return:
            eigval: a numpy vector of (m,)
            eigvec: a matrix of (m, m), column ``eigvec[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigval[i]``
            Note: eigval is not necessarily ordered
        """

        eigval, eigvec = np.linalg.eig(c)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)
        return eigval, eigvec

    def find_k(self, sorted_eig_vals):
        # Find total variance multiplied by retain ratio
        threshold = sum(sorted_eig_vals) * self.retain_ratio

        # Add to a running sum until the sum is greater than the threshold
        count = 0
        running_sum = 0
        while (running_sum < threshold):
            running_sum += sorted_eig_vals[count]
            count += 1

        # Return calculated k
        return count



    def fit(self, x):
        """
        fits the data x into the PCA. It results in self.eig_vecs and self.eig_values which will
        be used in the transform method
        :param x: input data of shape (n, m); n instances and m features
        :return:
            sets proper values for self.eig_vecs and eig_values
        """

        self.eig_vals = None
        self.eig_vecs = None

        x = x - PCA.mean(x)

        ########################################
        #       YOUR CODE GOES HERE            #
        ########################################
        #based this entirely on doc strings and input shapes :)
        covar = PCA.cov(x)
        self.eig_vals, self.eig_vecs = PCA.eig(covar)

        # sort eigen values and vectors : greatest to least

        """
        print("unsorted")
        print (self.eig_vecs)
        print (self.eig_vals)
        """

        sorted_eig_vecs = self.eig_vecs[self.eig_vals.argsort()]
        sorted_eig_vals = np.sort(self.eig_vals)

        sorted_eig_vecs = sorted_eig_vecs[::-1]
        sorted_eig_vals = sorted_eig_vals[::-1]


        """
        print("sorted")
        print (sorted_eig_vecs)
        print (sorted_eig_vals)
        """

        k = self.find_k(sorted_eig_vals)

        print("dims: ", self.eig_vecs.shape)

        # Only take the biggest k eigan value vector pairs
        self.eig_vecs = sorted_eig_vecs.T[:k].T
        self.eig_vals = sorted_eig_vals.T[:k].T

        print("post-slice dims: ", self.eig_vecs.shape)


    def transform(self, x):
        """
        projects x into lower dimension based on current eig_vals and eig_vecs
        :param x: input data of shape (n, m)
        :return: projected data with shape (n, len of eig_vals)
        """

        if isinstance(x, np.ndarray):
            x = np.asarray(x)
        if self.eig_vecs is not None:
            return np.matmul(x, self.eig_vecs)
        else:
            return x
