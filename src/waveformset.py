import numpy as np
import h5py
import waveform as wo #as in waveform object to avoid conflict with 'wave'


################################################################################
#                            HDF5WaveformSet class                             #
################################################################################

# TODO: implement __iter__ (which requires __next__),
# so you can do things like for h in training_set...
# __getitem__ doesn't know how to stop at last waveform.

class HDF5WaveformSet(object):
    """Methods for reading and writing a set of Waveform objects
    from an hdf5 file.

    Attributes
    ----------
    ws_file : h5py.File object
        The hdf5 file containing all the data.
    """

    ################## Methods to initialize and close object ################

    def __init__(self, filename, mode='a'):
        """Get a list of waveforms and data associated with those waveforms.

        Parameters
        ----------
        filename : string
            Name of the hdf5 file to store the HDF5WaveformSet
        #parameters : (n+1)darray
        #    Shape of parameter array containing waveforms each with array of parameters
        """
        # Open file if it exists. Create file if it doesn't already exist.
        self.ws_file = h5py.File(filename, mode, libver='latest')

    def close(self):
        """Close the hdf5 file.
        """
        self.ws_file.close()

    ###################### set and get Waveform data ########################

    def set_waveform(self, index, waveform, parameters):
        """Add a waveform to the hdf5 file.

        Parameters
        ----------
        waveform : Waveform
        parameters : List
            Parameters of the waveform
        index : int
            For index i, the waveform will be stored under the group 'wave_i'.
        """
        # The group name
        groupname = 'wave_'+str(index)

        # Delete the group if it already exists (so you can owerwrite it)
        if groupname in self.ws_file.keys():
            del self.ws_file[groupname]

        # Now write the data
        wave = self.ws_file.create_group(groupname)
        wave['parameters'] = parameters
        for key in waveform.data.keys():
            wave[key] = waveform.data[key]

    def get_waveform(self, index, data='waveform'):
        """Load a single Waveform object from the HDF5 file (or its waveform parameters).

        Parameters
        ----------
        index : int
            Index of the waveform you want.
        data : str, {'waveform', 'parameters'}
            The data to extract for the waveform.

        Returns
        -------
        Waveform for 'waveform'
        array for 'parameters'
        """
        # Get the waveform group
        groupname = 'wave_'+str(index)
        wave = self.ws_file[groupname]

        if data == 'waveform':
            # Create blank dictionary for waveform
            # Then fill it with the arrays in the wave group
            data = {}
            for key in wave.keys():
                if key != 'parameters':
                    data[key] = wave[key][:]
            return wo.Waveform(data)
        elif data == 'parameters':
            return wave['parameters'][:]
        else:
             raise Exception, "Valid data options for 'data' are {'waveform', 'parameters'}."

    def __getitem__(self, i):
        """Get waveform i using index notation: waveformset[i]
        """
        return self.get_waveform(i)

    ################### Properties of the HDF5WaveformSet #################

    def __len__(self):
        """The Number of waveforms in the HDF5WaveformSet.
        """
        names = self.ws_file.keys()
        wavegroups = [names[i] for i in range(len(names)) if 'wave_' in names[i]]
        return len(wavegroups)

    def parameters(self):
        """Get a list of the waveform parameters.

        Returns
        -------
        parameters : 2d list
            List of waveform parameters.
        """
        nwave = len(self)
        return np.array([self.get_waveform(i, data='parameters') for i in range(nwave)])


################ Functions for constructing a HDF5WaveformSet ##################


def waveform_set_from_list(filename, h_list, params_list):
    """Generate a HDF5WaveformSet from a list of waveforms and their parameters.
    """
    h_set = HDF5WaveformSet(filename)
    # Write the waveforms to h_set
    for i in range(len(h_list)):
        h_set.set_waveform(i, h_list[i], params_list[i])
    h_set.close()


def join_waveform_sets(filename1, filename2, filename_join):
    """Join two HDF5WaveformSets together into a new HDF5WaveformSet.
    """
    h_set1 = HDF5WaveformSet(filename1)
    h_set2 = HDF5WaveformSet(filename2)
    h_join = HDF5WaveformSet(filename_join)

    j = 0
    # Write the waveforms in h_set1 to h_join
    for i in range(len(h_set1)):
        h = h_set1.get_waveform(i, data='waveform')
        p = h_set1.get_waveform(i, data='parameters')
        h_join.set_waveform(j, h, p)
        j += 1

    # Append the waveforms in h_set2 to h_join
    for i in range(len(h_set2)):
        h = h_set2.get_waveform(i, data='waveform')
        p = h_set2.get_waveform(i, data='parameters')
        h_join.set_waveform(j, h, p)
        j += 1

    h_set1.close()
    h_set2.close()
    h_join.close()
