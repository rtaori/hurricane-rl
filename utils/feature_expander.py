import numpy as np
from mpl_toolkits.basemap import Basemap # NOTE: this needs to be installed
                                         # with "conda install basemap", not pip

class HurricaneFeatureExpander:
    """
    Hurricane Feature Expander

    Class that expands a data matrix with hurricane data with new features.

    """

    def __init__(self, data_matrix):
        """Initializes the Hurricane Feature Expander with a data matrix."""
        self.data_matrix = data_matrix
        self.bm = Basemap()

    def get_data_matrix(self):
        """Returns the data matrix at the current state."""
        return self.dataframe

    def add_is_land(self):
        """
        Adds a binary feature of whether the hurricane is on land (1) or on sea
        (0), for each hurricane sample.

        Assumes `lat` is column 0 and `lon` is column 1 in the data matrix.
        """
        is_land_feature = [self.bm.is_land(x[0], x[1]) for x in self.data_matrix]
        is_land_feature = np.array(is_land_feature).reshape(-1,1)
        np.hstack((self.data_matrix, is_land_feature))

    def add_temperature(self):
        """
        Adds the temperature feature to each sample in the data matrix. It uses
        the lat, lon, and time values to figure out the temperature at the time
        and location of the hurricane.
        """
        pass # Not implemented yet
