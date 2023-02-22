from jobshop.heuristic.operations import Graph


def get_times_Q(graph: Graph):
    """Get initial set of nodes to evaluate release-times
    """
    Q = []
    for m, machine in graph.M.items():
        jobs = machine.jobs
        j = jobs[0]
        if m == graph.seq[j][0]:
            Q.append((m, j))
    return Q


def get_tail_Q(graph: Graph):
    """Get initial set of nodes to evaluate tail-times
    """
    Q = []
    for m, machine in graph.M.items():
        jobs = machine.jobs
        j = jobs[-1]
        if m == graph.seq[j][-1]:
            Q.append((m, j))
    return Q


def calc_makespan(graph: Graph):
    """Evaluate release times based on pre-ordered graph structure

    Parameters
    ----------
    graph : Graph
        Graph structure of job-shop problem

    Returns
    -------
    float
        Makespan
    """
    
    Q = get_times_Q(graph)
    k = 0
    L = []
    C = 0

    while (len(Q) > 0) and (k <= 100000):
        
        # Add new item do labeled
        m, j = Q.pop(0)
        L.append((m, j))
        d = graph.O[m, j].duration
        r = 0
        
        # Get preceding and successors
        PJ = graph.precede_job(m, j)
        PM = graph.precede_machine(m, j)
        SJ = graph.follow_job(m, j)
        SM = graph.follow_machine(m, j)
        
        # Update start is preceded by any operations
        if PJ is not None:
            r = max(r, PJ.release + PJ.duration)
        if PM is not None:
            r = max(r, PM.release + PM.duration)
        graph.O[m, j].release = r
        
        # Update C if job is final
        if r + d >= C:
            C = r + d
        
        # Get successive jobs and add based on rule
        if SJ is not None:
            PM_SJ = graph.precede_machine(*SJ.code)
            if (PM_SJ is None) or (PM_SJ.code in L):
                if SJ.code not in Q:
                    Q.append(SJ.code)
        
        if SM is not None:
            PJ_SM = graph.precede_job(*SM.code)
            if (PJ_SM is None) or (PJ_SM.code in L):
                if SM.code not in Q:
                    Q.append(SM.code)
        
        # Add on k
        k = k + 1
    
    # Redefine C if there's a subcycle (infeasible)
    if len(L) < len(graph.O):
        C = graph.V * (1 + 1e-2)
    
    # Label graph C
    graph.C = C
    
    return C


def calc_tails(graph: Graph):
    """Evaluate tail times based on pre-ordered graph structure

    Parameters
    ----------
    graph : Graph
        Graph structure of job-shop problem

    Returns
    -------
    Graph
        Problem graph re-evaluated (also modified in place)
    """
    
    Q = get_tail_Q(graph)
    k = 0
    L = []

    while (len(Q) > 0) and k <= 100000:
        
        # Add new item do labeled
        m, j = Q.pop(0)
        L.append((m, j))
        d = graph.O[m, j].duration
        q = 0
        
        # Get preceding and successors
        PJ = graph.precede_job(m, j)
        PM = graph.precede_machine(m, j)
        SJ = graph.follow_job(m, j)
        SM = graph.follow_machine(m, j)
        
        # Update start is preceded by any operations
        if SJ is not None:
            q = max(q, SJ.tail)
        if SM is not None:
            q = max(q, SM.tail)
        graph.O[m, j].tail = q + d
        
        # Get successive jobs and add based on rule
        if PJ is not None:
            SM_PJ = graph.follow_machine(PJ.machine, PJ.job)
            if (SM_PJ is None) or ((SM_PJ.machine, SM_PJ.job) in L):
                Q.append((PJ.machine, PJ.job))
        
        if PM is not None:
            SJ_PM = graph.follow_job(PM.machine, PM.job)
            if (SJ_PM is None) or ((SJ_PM.machine, SJ_PM.job) in L):
                Q.append((PM.machine, PM.job))
        
        # Add on k
        k = k + 1
    
    return graph