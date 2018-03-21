from NonDominatedSort import NonDominatedSort

def NSGA(data, objective_funcs, pop_size, generations):

	assert len(objective_funcs) > 1, \
			"You need at least 2 objective functions."
	assert pop_size > 10, "You need at least 10 individuals"
	assert generations > 5, "You need at least 5 generations"