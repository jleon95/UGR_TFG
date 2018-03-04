import numpy as np

def InitializeFirstFront(objective_scores):

	# "individuals" contains pairs in the format:
	# 1st element: number of individuals that dominate this one
	# 2nd element: list of individuals dominated by this one
	individuals = []
	fronts = np.zeros(objective_scores.shape[0])
	n_objectives = objective_scores.shape[1]

	for i in range(objective_scores.shape[0]): # For each individual

		individuals.append([0,[]])

		for j in range(objective_scores.shape[0]): # For every other individual

			dom_less = 0
			dom_equal = 0
			dom_more = 0

			for k in range(n_objectives): # For every objective function value

				# Remember that smaller values are better, the optimal being 0
				if objective_scores[i,k] < objective_scores[j,k]: 
					dom_less += 1
				elif objective_scores[i,k] == objective_scores[j,k]:
					dom_equal += 1
				else
					dom_more += 1

			if dom_less == 0 && dom_equal != n_objectives: # i is farther from the origin than j
				individuals[-1][0] += 1 # Acknowledge that j dominates i
			elif dom_more == 0 && dom_equal != n_objectives # i is closer to the origin than j
				individuals[-1][1].append(j) # Acknowledge that i dominates j (index-based)

		if individuals[-1][0] == 0:
			fronts[i] = 1 # If no other individual dominates i, it belongs to front 1

	return individuals, fronts

# "front info"  is a list containing the information about domination between individuals
# that was obtained in InitializeFirstFront.
# "front_scores" is a numpy array containing the fronts found in InitializeFirstFront,
# which means 1 if in first front or default 0 if yet to know.
def FillFronts(front_info, front_scores):

	front_number = 1
	current front = []
	for i in range(front_scores.shape[0]):

		if front_scores[i] == 1: # If the ith individual is in front 1
			current_front.append(i) # add it to this temporary list.

	while current_front: # While there are individuals without assigned front

		next_front = []
		for i in current_front: # For each individual in this front

			if front_info[i][1]: # If i dominates any other individual

				for j in front_info[i][1]: # Loop over those individuals

					front_info[j][0] -= 1
					if front_info[j][0] == 0 # If no other individual dominates j,
						front_scores[j] = front + 1 # it belongs to front n+1.
						next_front.append(j)
						
		front += 1
		current_front = list(next_front)

	return front_scores # Now we should have a front number for every individual

# "objective_scores" consists of a matrix of n_individuals x n_objectives
# containing the scores of the individuals of a population for certain
# optimization objectives.
def NonDominatedSort(objective_scores):

	# "sort scores" will contain pairs of [front number, crowding distance] for each individual
	# in the original order of appearance in "population".
	sort_scores = np.zeros((objective_scores.shape[0],2))
	front_info, sort_scores[:,0] = InitializeFirstFront(objective_scores)
	sort_scores[:,0] = FillFronts(front_info,sort_scores[:,0])