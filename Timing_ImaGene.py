import os
import gzip
import numpy as np

class ImaGene:
    def __init__(self, data, positions, description=[], targets=[], parameter_name=None, classes=[]):
        self.data = data
        self.positions = positions
        self.description = description
        self.dimensions = (np.zeros(len(self.data)), np.zeros(len(self.data)))
        # initialise dimensions to the first image (in case we have only one)
        self.dimensions[0][0] = self.data[0].shape[0]
        self.dimensions[1][0] = self.data[0].shape[1]
        # if reads from real data, then stop here otherwise fill in all info on simulations
        if parameter_name != None:
            self.parameter_name = parameter_name # this is passed by ImaFile.read_simulations()
            self.targets = np.zeros(len(self.data), dtype='int32')
            for i in range(len(self.data)):
                # set targets from file description
                self.targets[i] = self.description[i][self.parameter_name]
                # assign dimensions
                self.dimensions[0][i] = self.data[i].shape[0]
                self.dimensions[1][i] = self.data[i].shape[1]
            self.classes = np.unique(self.targets)
        return None

    def read_simulations(self, parameter_name='selection_start_time', max_nrepl=None, verbose=0):
        """
        Read simulations and store into compressed numpy arrays

        Keyword Arguments:
            parameter_name: name of parameter to estimate
            max_nrepl: max nr of replicates per simulated msms file
            verbose: 

        Returns:
            an object of class Genes
        """

        data = []
        positions = []
        description = []

        # Open the directory in which simulation files are stored
        for file_name in os.listdir(self.simulations_folder):

            full_name = self.simulations_folder + '/%s' %(file_name)

            if verbose > 0:
                print(full_name, ': ', end='')

            # Read lines including the metadata
            f = gzip.open(full_name, 'rb')
            file_content = f.read().decode('utf8').split('\n')

            # Search the // char inside the file
            starts = ([i for i, e in enumerate(file_content) if e == '//'])

            # limit the scan to the first max_nrepl items (if set)
            if max_nrepl!=None:
                starts = starts[:max_nrepl]

            if verbose > 0:
                print(len(starts))

            # Populate object with data for each simulated gene
            for idx, pointer in enumerate(starts):

                # Description for each simulation
                description.append(self.extract_description(full_name, file_content[0]))

                nr_columns = int(file_content[pointer+1].split('segsites: ')[1])
                haplotypes = np.zeros((self.nr_samples, nr_columns, 1), dtype='uint8')
                pos = file_content[pointer+2].split(' ')
                pos.pop()
                pos.pop(0)
                positions.append(np.asarray(pos, dtype='float32'))
                del pos

                for j in range(self.nr_samples):

                    hap = list(file_content[pointer + 3 + j])

                    # string processing: if not 0/1 --> convert to 1
                    hap = ['1' if element!='0' and element!=1 else element for element in hap]
                    # switch colours, 1s are black and 0s are white
                    hap = ['255' if element=='1' else element for element in hap]
                    haplotypes[j,:,0] = hap

                data.append(haplotypes)

            f.close()

        gene = ImaGene(data=data, positions=positions, description=description, parameter_name=parameter_name)
        return gene

    def set_targets(self):
        """
        Set targets for binary classification based on selection start time AFTER running set_classes
        """
        # Initialise
        self.targets = np.zeros(len(self.data), dtype='int32')
        for i in range(len(self.targets)):
            # Reinitialise
            selection_start_time = self.description[i]['selection_start_time']
            # Assign label based on timing of selection
            self.targets[i] = 0 if selection_start_time <= 0.05 else 1  # Classify 0 for recent (<= 50kya), 1 for ancient (> 50kya)
        return 0

    def set_classes(self, classes=[], nr_classes=0):
        """
        Set classes (or reinitiate)
        """
        # at each call reinitialise for safety
        targets = np.zeros(len(self.data), dtype='int32')
        for i in range(len(self.data)):
            # set target from file description
            targets[i] = self.description[i][self.parameter_name]
        self.classes = np.unique(targets)
        # calculate and/or assign new classes
        if nr_classes > 0:
            self.classes = np.asarray(np.linspace(targets.min(), targets.max(), nr_classes), dtype='int32')
        elif len(classes)>0:
            self.classes = classes
        del targets
        return 0

    def extract_description(self, file_name, first_line):
        """
        Read first line of simulations, extract all metadata and store it in a dictionary

        Keyword Arguments:
            file_name (string) -- name of simulation file
            first_line (string) -- first line of gzipped msms file
            model_name (string) -- name of demographic model

        Return:
            description (string)
        """

        desc = {'name':file_name}

        # Extracting parameters
        desc.update({'Nref':int(extract_msms_parameter(first_line, '-N '))})
        desc.update({'nr_chroms':int(extract_msms_parameter(first_line, '-N ', 1))})
        desc.update({'nr_replicates':int(extract_msms_parameter(first_line, '-N ', 2))})

        desc.update({'mutation_rate':float(extract_msms_parameter(first_line, '-t '))})
        desc.update({'recombination_rate':float(extract_msms_parameter(first_line, '-r '))})
        desc.update({'recombination_rate_nr_sites':int(extract_msms_parameter(first_line, '-r ', 1))})

        desc.update({'selection_position':float(extract_msms_parameter(first_line, '-Sp '))})
        desc.update({'selection_start_time':float(extract_msms_parameter(first_line, '-SI ', 0))})
        desc.update({'selection_start_frequency':float(extract_msms_parameter(first_line, '-SI ', 2))})
    
        desc.update({'selection_coeff_HOMO':int(extract_msms_parameter(first_line, '-SAA '))})
        desc.update({'selection_coeff_hetero':int(extract_msms_parameter(first_line, '-SAa '))})
        desc.update({'selection_coeff_homo':int(extract_msms_parameter(first_line, '-Saa '))})

        desc.update({'model':str(self.model_name)})

        # Get the UNIX Time Stamp of when the file was modification
        desc.update({'modification_stamp':os.stat(file_name).st_mtime})

        # Allow deleted files to be tracked in json folder
        desc.update({'active':'active'})

        return desc

def extract_msms_parameter(line, parameter, position=0):
    """
    Extract parameter from msms command line

    Keyword Arguments:
        line: msms command line
        parameter: name of parameter to extract
        position: position of parameter to extract

    Returns:
        value of parameter
    """
    elements = line.split()
    for i, elem in enumerate(elements):
        if elem == parameter:
            return elements[i + 1 + position]
    return None

# Example usage:
# gene_sim = ImaGene(data, positions, description, parameter_name='selection_start_time')
# gene_sim.set_targets()
