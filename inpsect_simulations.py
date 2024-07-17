import os
import gzip

def extract_parameters(line):
    """Extract parameters from a given line."""
    params = {
        'Nref': extract_msms_parameter(line, '-N '),
        'nr_chroms': extract_msms_parameter(line, '-N ', 1),
        'nr_replicates': extract_msms_parameter(line, '-N ', 2),
        'mutation_rate': extract_msms_parameter(line, '-t '),
        'recombination_rate': extract_msms_parameter(line, '-r '),
        'recombination_rate_nr_sites': extract_msms_parameter(line, '-r ', 1),
        'selection_position': extract_msms_parameter(line, '-Sp '),
        'selection_start_time': extract_msms_parameter(line, '-SI '),
        'selection_start_frequency': extract_msms_parameter(line, '-SI ', 2),
        'selection_coeff_HOMO': extract_msms_parameter(line, '-SAA '),
        'selection_coeff_hetero': extract_msms_parameter(line, '-SAa '),
        'selection_coeff_homo': extract_msms_parameter(line, '-Saa ')
    }
    return params

def extract_msms_parameter(line, param, index=0):
    """Extract the value of a parameter from the line."""
    try:
        start = line.index(param) + len(param)
        end = line.find(' ', start)
        if end == -1:
            end = len(line)
        parts = line[start:end].split()
        if len(parts) > index:
            return parts[index]
        else:
            return None
    except ValueError:
        return None

def inspect_simulation_files(simulations_path):
    """Inspect the first line of each simulation file in the given path."""
    for root, dirs, files in os.walk(simulations_path):
        for file in files:
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline().strip()
                    print(f"First line of {file}: {first_line}")
                    params = extract_parameters(first_line)
                    print(f"Parameters: {params}")

if __name__ == "__main__":
    simulations_path = "/data/home/ha231431/EvoNet-CNN-Insight/AM"  # Adjust path as necessary
    inspect_simulation_files(simulations_path)

