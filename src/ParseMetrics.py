# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#	By Javier León Palomares, University of Granada, 2018   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import csv
import re

# Parses the human-readable output of the genetic algorithms
# and saves the metric loss evolution to individual files.
def ParseMetrics(metrics_file, output_prefix):

	lines = open(metrics_file).readlines()
	individuals = [x for x,y in enumerate(lines) if "Individual" in y]

	for i in individuals:

		number = lines[i].split()[1]

		with open(output_prefix+number,"w") as output:

			j = i+9
			values = re.findall("\d.\d+",lines[j])

			while values:

				output.write(" ".join(values)+"\n")
				j += 1
				values = re.findall("\d.\d+",lines[j])

