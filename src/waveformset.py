import numpy as np
import h5py
import waveform as wo #as in waveform object to avoid conflict with 'wave'


################################################################################
#                            HDF5WaveformSet class                             #
################################################################################

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
