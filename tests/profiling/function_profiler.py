from sola.main_classes import functions
from sola.main_classes import domains
import numpy as np
from line_profiler import LineProfiler
import sys


if __name__ == "__main__":

    domain = domains.HyperParalelipiped([[-1, 1]])
    # Define a function to execute your method within the class context
    r = np.linspace(0, 1, 1000000)
    instances = [functions.Null_1D(domain=domain),
                 functions.Constant_1D(domain=domain, value=1),
                 functions.Gaussian_Bump_1D(domain=domain, center=0, width=0.5), # noqa
                 functions.Dgaussian_Bump_1D(domain=domain, center=0, width=0.5), # noqa
                 functions.Gaussian_1D(domain=domain, center=0, width=0.5),
                 functions.Moorlet_1D(domain=domain, center=0, spread=0.5, frequency=1), # noqa
                 functions.Haar_1D(domain=domain, center=0, width=0.5),
                 functions.Ricker_1D(domain=domain, center=0, width=0.5),
                 functions.Dgaussian_1D(domain=domain, center=0, width=0.5),
                 functions.Boxcar_1D(domain=domain, center=0, width=0.5),
                 functions.Bump_1D(domain=domain, center=0, width=0.5),
                 functions.Triangular_1D(domain=domain, center=0, width=0.5),
                 functions.NormalModes_1D(domain=domain, order=2, spread=1, max_freq=1, seed=1)] # noqa

    # Create a LineProfiler object
    profiler = LineProfiler()

    # Add the method to profile to the LineProfiler object
    for instance in instances:
        profiler.add_function(instance.evaluate)

    # Run the profiler
    profiler.enable()
    for instance in instances:
        instance.evaluate(r)
    profiler.disable()

    # Print the profiling results
    with open('function_profiler_results.txt', 'w') as f:
        sys.stdout = f
        profiler.print_stats(output_unit=1e-9)
        sys.stdout = sys.__stdout__

    # Collect and compare total times
    # Open the file in read mode and read lines into a list
    with open('function_profiler_results.txt', 'r') as file:
        lines = file.readlines()

        # Initialize an empty list to hold the total times
        total_times = []

        # Iterate over the lines
        for line in lines:
            # If the line starts with 'Total time:', extract the time and add
            # it to the list
            if line.startswith('Total time:'):
                # split the line at the colon and remove leading/trailing
                # whitespace
                time = float(line.split(':')[1].split()[0].strip())
                total_times.append(time)

        minimum_time = min(total_times)
        normalized_times = [str(time / minimum_time) for time in total_times]

        # Open the file in append mode and write the total times at the end
        with open('function_profiler_results.txt', 'a') as file:
            file.write('\nTotal times for each function:\n')
            for normalized_time, time, instance in zip(normalized_times,
                                                       total_times,
                                                       instances):
                file.write(str(instance) + ': ' + str(time) + ' | ' +
                           normalized_time + '\n')
