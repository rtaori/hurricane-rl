import os
import numpy as np
from mpl_toolkits.basemap import Basemap # NOTE: This needs to be installed
                                         # with "conda install basemap", not pip
from data_structure import TemperatureDictionary

class HurricaneFeatureExpander:
    """
    Hurricane Feature Expander

    Class that expands a data matrix with hurricane data with new features.

    """

    def __init__(self, data_matrix, temp_data):
        """
        Initializes the Hurricane Feature Expander with a data matrix.

        It assumes the first features are as follows: lat, lon, year, month.
        """
        self.data_matrix = data_matrix
        self.temp_data = temp_data
        self.bm = Basemap()

    def get_data_matrix(self):
        """Returns the data matrix at the current state."""
        return self.data_matrix

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

        Assumes `lat` is column 0, `lon` is column 1, `year` is column 2, and
        `month` is column 3 in the data matrix.
        """
        # 1. Create the temperature feature column
        temperature_feature = [self.temp_data.get(x) for x in self.data_matrix]
        temperature_feature = np.array(temperature_feature).reshape(-1,1)

        # 2. Append the temperature feature column
        np.hstack((self.data_matrix, temperature_feature))
