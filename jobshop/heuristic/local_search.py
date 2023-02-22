from jobshop.heuristic.operations import Graph
from jobshop.heuristic.evaluation import calc_tails, calc_makespan


def get_critical(graph: Graph):
    """Assign critical attribute of operations from graph"""
    for key, o in graph.O.items():
        if o.tail + o.release == graph.C:
            graph.O[key].critical = True
        else:
            graph.O[key].critical = False


def find_swaps(graph: Graph):
    """Obtains swap of nodes

    Parameters
    ----------
    graph : Graph
        Graph structure of job-shop problem

    Returns
    -------
    list of tuples
        Each tuple contains (m, i, j), in which m is a machine and i, j are two
        consecutive critical jobs that take place in m
    """
    swaps = []
    for m in graph.M:
        last_critical = False
        last_job = None
        for j in graph.M[m].jobs:
            if graph.O[m, j].critical:
                if last_critical:
                    swaps.append((m, last_job, j))
                last_critical = True
            else:
                last_critical = False
            last_job = j
    return swaps


def calc_cost(swap: tuple, graph: Graph):
    """Calculates the new cost function based on a move

    Parameters
    ----------
    swap : tuple
        In the format (m, i, j), in which m is a machine and i, j are two
        consecutive critical jobs that take place in m
    
    graph : Graph
        Graph structure of job-shop problem

    Returns
    -------
    float
        New makespan after move
    """
    
    # Label operations of swap
    a = graph.O[swap[0], swap[1]]
    b = graph.O[swap[0], swap[2]]
    
    # Find previous and following operations
    PJa = graph.precede_job(*a.code)
    PMa = graph.precede_machine(*a.code)
    SJa = graph.follow_job(*a.code)
    SMa = graph.follow_machine(*a.code)
    
    PJb = graph.precede_job(*b.code)
    PMb = graph.precede_machine(*b.code)
    SJb = graph.follow_job(*b.code)
    SMb = graph.follow_machine(*b.code)
    
    # Calc equation terms for preceding
    if PMa is None:
        PMa_term = 0.0
    else:
        PMa_term = PMa.release + PMa.duration
    if PJb is None:
        PJb_term = 0.0
    else:
        PJb_term = PJb.release + PJb.duration
    if PJa is None:
        PJa_term = 0.0
    else:
        PJa_term = PJa.release + PJa.duration
    
    # Calc equation terms for next
    if SMa is None:
        q_SMa = 0.0
    else:
        q_SMa = SMa.tail
    if SMb is None:
        q_SMb = 0.0
    else:
        q_SMb = SMb.tail
    if SJa is None:
        q_SJa = 0.0
    else:
        q_SJa = SJa.tail
    if SJb is None:
        q_SJb = 0.0
    else:
        q_SJb = SJb.tail
    
    # New releases
    rb_new = max(PMa_term, PJb_term)
    ra_new = max(rb_new + b.duration, PJa_term)
    
    # New tails
    qa_new = max(q_SMb, q_SJa) + a.duration
    qb_new = max(qa_new, q_SJb) + b.duration
    
    # New makespan
    C_new = max(rb_new + qb_new, ra_new + qa_new)
    
    return C_new


def find_best_move(graph: Graph):
    """Finds the best move from a neighborhood of a graph that describes
    a current state of the job-shop problem

    Parameters
    ----------
    graph : Graph
        Graph structure of job-shop problem

    Returns
    -------
    tuple
        Best move in the format (m, i, j), in which m is a machine and i, j are two
        consecutive critical jobs that take place in m
    """
    
    # Save current solution
    C_best = graph.C
    best_move = None
    swaps = find_swaps(graph)
    
    # Iterate over swaps
    if swaps is not None:
        for swap in swaps:
            C_swap = calc_cost(swap, graph)
            if C_swap < C_best:
                best_move = swap
                C_best = C_swap
    
    return best_move


def _local_search_step(graph, copy=False):
    
    # Define new graph
    if copy:
        new_graph = graph.copy()
    else:
        new_graph = graph
    
    # Obtain best move
    best_move = find_best_move(new_graph)
    
    # Do swaps if not None
    if best_move is not None:
        m, i, j = best_move
        new_graph.M[m].jobs.swap(i, j)
        
        # Re-calculate costs and tails
        calc_makespan(new_graph)
        calc_tails(new_graph)
        get_critical(new_graph)
    
    return new_graph


def local_search(graph: Graph, max_steps=1000, copy=False):
    """Do local search from current state of job-shop problem

    Parameters
    ----------
    graph : Graph
        Graph structure of job-shop problem
    
    max_steps : int, optional
        Maximum number of steps in local search, by default 1000
    
    copy : bool, optional
        Either or not to do deepcopy of the graph in each step, by default False

    Returns
    -------
    Graph
        Local optima graph
    """
    
    # Get current state
    C = graph.C
    proceed = True
    k = 0
    
    # Do local steps until no improvement is found
    while proceed and k < max_steps:
        graph = _local_search_step(graph, copy=copy)
        C_new = graph.C
        if C_new < C:
            C = C_new
        else:
            proceed = False
        k = k + 1
    
    return graph