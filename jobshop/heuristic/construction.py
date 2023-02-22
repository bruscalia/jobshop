import numpy as np
from jobshop.heuristic.operations import Graph


def get_constr_Q(graph: Graph):
    Q = []
    for j, machines in graph.seq.items():
        Q.append((machines[0], j))
    return Q


def semi_greedy_makespan(graph: Graph, alpha=0.9):
    """Do a semi-greedy construction based on incremental makespan

    Parameters
    ----------
    graph : Graph
        Problem in a graph structure
    
    alpha : float, optional
        Greedy criterion parameter, by default 0.9
    """
    
    # Get initial candidates
    Q = get_constr_Q(graph)
    k = 0
    max_iter = len(graph.jobs) * len(graph.machines) + 1

    # Iterate if there's additional jobs to include
    while len(Q) >= 1 and k <= max_iter:
        
        k = k + 1

        # Start 
        C = 0
        H = {}
        R = {}
        S = []

        # Iterate over feasible solutions
        for q in Q:
            
            # Add new item to labeled
            m, j = q
            d = graph.O[m, j].duration
            r = 0
            
            # Get preceding
            PJ = graph.precede_job(m, j)
            prev_jobs = graph.M[m].jobs
            if len(prev_jobs) >= 1:
                last_job = graph.M[m].jobs[-1]
                PM = graph.O[m, last_job]
            else:
                PM = None
            
            # Update minumum release
            if PJ is not None:
                r = max(r, PJ.release + PJ.duration)
            if PM is not None:
                r = max(r, PM.release + PM.duration)
            R[q] = r
            
            # Update C_pot
            C_pot = max(r + d, C)
            H[q] = C_pot
            
        # Get tolerance
        min_sol = min(H.values())
        max_sol = max(H.values())
        atol = min_sol + (1 - alpha) * (max_sol - min_sol)

        # Iterate over candidates
        for q, C_pot in H.items():
            if C_pot <= atol:
                S.append(q)
            else:
                pass
            
        # Add random element
        idx = np.random.choice(len(S))
        new_element = S[idx]
        m, j = new_element
        
        graph.O[m, j].release = R[m, j]
        graph.M[m].add_job(j)
        Q.pop(Q.index((m, j)))
        
        # Update C
        C = max(C, H[m, j])
        
        # Get next machine of job (now feasible)
        SJ = graph.follow_job(m, j)
        if SJ is not None:
            Q.append(SJ.code)


def get_job_total_duration(job, graph: Graph):
    total = 0
    for m in graph.seq[job]:
        total = total + graph.p_times[m, job]
    return total


def semi_greedy_time_remaining(graph: Graph, alpha=0.9):
    """Do a semi-greedy construction based on job remaining time

    Parameters
    ----------
    graph : Graph
        Problem in a graph structure
    
    alpha : float, optional
        Greedy criterion parameter, by default 0.9
    """
    
    # Get initial candidates
    Q = get_constr_Q(graph)
    time_remaining = {job: get_job_total_duration(job, graph) for job in graph.jobs}
    
    k = 0

    # Iterate if there's additional jobs to include
    while len(Q) >= 1 and k <= 10000:
        
        k = k + 1

        # Start
        H = {}
        S = []

        # Iterate over feasible solutions
        for q in Q:
            
            # Add new item to labeled
            m, j = q
            H[q] = - (time_remaining[j] - graph.O[m, j].duration)
            
        # Get tolerance
        min_sol = min(H.values())
        max_sol = max(H.values())
        atol = min_sol + (1 - alpha) * (max_sol - min_sol)

        # Iterate over candidates
        for q, d in H.items():
            if d <= atol:
                S.append(q)
            else:
                pass
            
        # Add random element
        idx = np.random.choice(len(S))
        new_element = S[idx]
        m, j = new_element
        graph.M[m].add_job(j)
        time_remaining[j] = time_remaining[j] - graph.O[m, j].duration
        Q.pop(Q.index((m, j)))
        
        # Get next machine of job (now feasible)
        SJ = graph.follow_job(m, j)
        if SJ is not None:
            Q.append(SJ.code)