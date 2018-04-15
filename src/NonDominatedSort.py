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
				else:
					dom_more += 1

			if dom_less == 0 and dom_equal != n_objectives: # i is farther from the origin than j
				individuals[-1][0] += 1 # Acknowledge that j dominates i
			elif dom_more == 0 and dom_equal != n_objectives: # i is closer to the origin than j
				individuals[-1][1].append(j) # Acknowledge that i dominates j (index-based)

		if individuals[-1][0] == 0:
			fronts[i] = 1 # If no other individual dominates i, it belongs to front 1

	return individuals, fronts

# "front info" is a list containing the information about domination between individuals
# that was obtained in InitializeFirstFront.
# "front_scores" is a numpy array containing the fronts found in InitializeFirstFront,
# which means 1 if in first front or default 0 if yet to know.
def FillFronts(front_info, front_scores):

	front_number = 1
	current_front = []
	for i in range(front_scores.shape[0]):

		if front_scores[i] == 1: # If the ith individual is in front 1
			current_front.append(i) # add it to this temporary list.

	while current_front: # While there are individuals without assigned front

		next_front = []
		for i in current_front: # For each individual in this front

			if front_info[i][1]: # If i dominates any other individual

				for j in front_info[i][1]: # Loop over those individuals

					front_info[j][0] -= 1
					if front_info[j][0] == 0: # If no other individual dominates j,
						front_scores[j] = front_number + 1 # it belongs to front n+1.
						next_front.append(j)
						
		front_number += 1
		current_front = list(next_front)

	return front_scores # Now we should have a front number for every individual

def CrowdingDistance(objective_scores, front_scores):

	# Indices for the individuals in fronts 1, 2, 3 and so on
	front_sort_indices = np.argsort(front_scores)
	max_front = int(front_scores[front_sort_indices[-1]])
	distances = np.zeros(objective_scores.shape[0])
	start_index = 0 # Points to the start of this front
	end_index = 0 # Points to the end of this front

	for front in range(1, max_front+1):

		# Find the interval of individuals of this front
		while (end_index < front_sort_indices.shape[0] and
			front_scores[front_sort_indices[end_index]] == front):

			end_index += 1

		# The objective scores for the individuals of this front 
		objective_scores_front = objective_scores[front_sort_indices[start_index:end_index]]

		for obj in range(objective_scores.shape[1]):

			# Sort the new subset by their score at objective "obj"
			obj_sort_indices = np.argsort(objective_scores_front[:,obj])
			sorted_front = objective_scores_front[obj_sort_indices,obj]

			f_max = sorted_front[-1]
			f_min = sorted_front[0]

			# Boundary values (i.e. the first and the last) have
			# an infinite value of distance.
			# "start_index" is added because "obj_sort_indices" is a local sort
			# on the array of individuals of a front, meaning that there's an 
			# offset when considering the whole array.
			distances[front_sort_indices[obj_sort_indices[0]+start_index]] = float("inf")
			distances[front_sort_indices[obj_sort_indices[-1]+start_index]] = float("inf")

			# Now we calculate the values that are inside the boundaries
			for i in range(1, obj_sort_indices.shape[0]-1):

				if f_max - f_min == 0:
					distance[front_sort_indices[obj_sort_indices[i]+start_index]] = float("inf")
				else:
					next_value = sorted_front[i+1]
					prev_value = sorted_front[i-1]
					distances[front_sort_indices[obj_sort_indices[i]+start_index]] += \
						(next_value - prev_value) / (f_max - f_min)

		start_index = end_index

	return distances

# "objective_scores" consists of a matrix of n_individuals x n_objectives
# containing the scores of the individuals of a population for certain
# optimization objectives.
def NonDominatedSortScores(objective_scores):

	# "sort scores" will contain pairs of [front number, crowding distance] for each individual
	# in the original order of appearance in "population".
	sort_scores = np.zeros((objective_scores.shape[0],2))
	front_info, sort_scores[:,0] = InitializeFirstFront(objective_scores)
	sort_scores[:,0] = FillFronts(front_info,sort_scores[:,0])
	sort_scores[:,1] = CrowdingDistance(objective_scores,sort_scores[:,0])
	return sort_scores

# Assuming non-dominated sort scores with columns [front, crowding_distance],
# sorts according to two keys:
# - Front: ascending order, primary criterion.
# - Crowding distance: descending order, secondary criterion.
# Returns the indices that sort the elements.
def IndirectSort(sort_scores):

	return np.lexsort((-sort_scores[:,1],sort_scores[:,0]))