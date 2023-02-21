from jobshop.heurstic.operations import Graph
from jobshop.heurstic.evaluation import calc_makespan


def get_delta_solutions(S: Graph, T: Graph):
    delta_set = {}
    for m in S.machines:
        delta_set[m] = []
        for k, j in enumerate(S.M[m].jobs):
            i = T.M[m].jobs[k]
            if i != j:
                if (j, i) not in delta_set[m]:
                    delta_set[m].append((i, j))
    return delta_set


def path_relinking(S: Graph, T: Graph, max_iter=1000):
    
    # Initialize values
    c_gmin = S.C
    S_gmin = S.copy()
    delta_sol = get_delta_solutions(S, T)
    iter_count = 0
    total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
    
    # Do path
    while total_lenght >= 2 and iter_count <= max_iter:
        
        # Initialize values of iteration
        c_min = float("inf")
        
        # Iterate over machines
        for m in S.machines:
            # Iterate over swaps of machine
            for (i, j) in delta_sol[m]:
                S_alt = S.copy()
                S_alt.M[m].jobs.swap(i, j)
                c_alt = calc_makespan(S_alt)
                
                # If better than previous update
                if c_alt <= c_min:
                    c_min = c_alt
                    S_min = S_alt
                    best_swap = (i, j)
                    m_min = m
        
        # Update after move
        S = S_min
        makespan = c_min
        delta_sol[m_min].remove(best_swap)
        total_lenght = sum(len(delta_machine) for delta_machine in delta_sol.values())
        
        # Update global best
        if makespan <= c_gmin:
            c_gmin = makespan
            S_gmin = S
         
        # Update iterations
        iter_count = iter_count + 1
    
    return S_gmin