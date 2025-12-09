from queue import Queue
import psutil
import os
import time
import pm4py
import dd.autoref as bdd

# ILP
import pulp

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []

    def __str__(self):
        result = ["=== Petri Net Info ==="]

        result.append("\nPlaces:")
        for pid, tokens in self.places.items():
            result.append(f"  {pid}: tokens = {tokens}")

        result.append("\nTransitions:")
        for tid, name in self.transitions.items():
            result.append(f"  {tid}: name = {name}")

        result.append("\nArcs:")
        for arc in self.arcs:
            result.append(f"  {arc[0]} -> {arc[1]} , weight : {arc[2]}")

        result.append("=====================")
        return "\n".join(result)

   
# task 1


def check_consistency(net : PetriNet) -> bool:
    ok = True

    # must have places & transitions
    if len(net.places) == 0:
        print("No places in net")
        ok = False
    if len(net.transitions) == 0:
        print("No transitions in net")
        ok = False

    # duplicate check
    if len(net.places) != len(set(net.places.keys())):
        print("Duplicate place IDs detected")
        ok = False
    if len(net.transitions) != len(set(net.transitions.keys())):
        print("Duplicate transition IDs detected")
        ok = False

    # token & 1-safe check
    for p, tok in net.places.items():
        if not isinstance(tok, int):
            print(f"Marking value for place {p} is not integer")
            ok = False
        if tok < 0:
            print(f"Negative tokens at place {p}")
            ok = False
        if tok > 1:
            print(f"[1-safe violated] Place {p} has {tok} tokens (must be <= 1)")
            ok = False

    # arc validation
    place_ids = set(net.places.keys())
    trans_ids = set(net.transitions.keys())
    for s, t, w in net.arcs:
        if s not in place_ids and s not in trans_ids:
            print(f"Arc source '{s}' does not exist")
            ok = False
        if t not in place_ids and t not in trans_ids:
            print(f"Arc target '{t}' does not exist")
            ok = False
        if w != 1:
            print(f"[Invalid weight] Arc {s} -> {t} has weight {w} (must be 1)")
            ok = False

    if ok:
        print("PNML consistency check passed!")
    else:
        print("PNML consistency check failed!")

    return ok


def read_pnmlFile(filepath: str) -> PetriNet:
    net_pm4py, initial_marking, final_marking = pm4py.read_pnml(filepath)

    net = PetriNet()
    # get places
    for place in net_pm4py.places:
        id = place.name
        token = initial_marking[place] if place in initial_marking else 0
        net.places[id] = token
    # get transitions
    for transition in net_pm4py.transitions:
        trans_id = transition.name
        label = transition.label if transition.label is not None else transition.name
        net.transitions[trans_id] = label
    # get arcs    
    for arc in net_pm4py.arcs:
        source_id = arc.source.name
        target_id = arc.target.name
        weight = arc.weight if hasattr(arc, 'weight') and arc.weight is not None else 1
        net.arcs.append((source_id, target_id, weight))
    # check consistency
    if(check_consistency(net)):
        return net
    else:
        exit();

# task 2

def all_reachable_marking(net: PetriNet) -> list[dict]:
    # Build input/output structures
    inp = {}
    outp = {}
    inp_w = {}
    outp_w = {}

    for source, target, w in net.arcs:
        if source in net.places:
            inp.setdefault(target, []).append(source)
            inp_w[(source, target)] = w
        elif source in net.transitions:
            outp.setdefault(source, []).append(target)
            outp_w[(source, target)] = w

    # Initial marking
    initial = {p: int(count) for p, count in net.places.items()}
    visited = [initial]

    # BFS exploration
    queue = Queue()
    queue.put(initial)

    while not queue.empty():
        marking = queue.get()
        for tran in net.transitions:
            tran_in = inp.get(tran, [])
            tran_out = outp.get(tran, [])

            # Check if transition is enabled
            fire = True
            for p in tran_in:
                if marking[p] < inp_w.get((p, tran), 1):
                    fire = False
                    break


            if fire:
                for p in tran_out:
                    if p not in tran_in and marking[p] > 0:
                        fire = False
                        break

            if fire:
                new_marking = dict(marking)
                # Consume tokens from input places
                for p in tran_in:
                    new_marking[p] -= inp_w[(p, tran)]
                # Produce tokens to output places
                for p in tran_out:
                    new_marking[p] += outp_w.get((tran, p), 1)

                if new_marking not in visited:
                    visited.append(new_marking)
                    queue.put(new_marking)

    return visited

# task 3

def bbd(net: PetriNet, verbose: bool = False):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Create BDD manager
    manager = bdd.BDD()

    # Sort places for consistent ordering
    places_list = sorted(net.places.keys())
    n = len(places_list)

    # Create variables: current and next states
    curr_vars = [f"c_{i}" for i in range(n)]
    next_vars = [f"n_{i}" for i in range(n)]

    all_vars = curr_vars + next_vars
    manager.declare(*all_vars)

    # Create BDD for initial marking
    initial_marking = {p: int(net.places[p]) for p in places_list}
    R = manager.true

    for i, place in enumerate(places_list):
        var = curr_vars[i]
        if initial_marking[place] == 1:
            R &= manager.add_expr(var)
        else:
            R &= manager.add_expr(f"~{var}")

    # Build transition relation T
    # Precompute input/output places for each transition
    inp = {}
    outp = {}
    for source, target, w in net.arcs:
        if source in net.places and target in net.transitions:
            inp.setdefault(target, []).append(source)
        elif source in net.transitions and target in net.places:
            outp.setdefault(source, []).append(target)

    # Start with false, accumulate transitions
    T = manager.false

    for tran in net.transitions:
        tran_in = inp.get(tran, [])
        tran_out = outp.get(tran, [])

        # Get indices for places
        in_indices = [places_list.index(p) for p in tran_in]
        out_indices = [places_list.index(p) for p in tran_out]

        # Build transition BDD for this transition
        tran_bdd = manager.true

        # Condition 1: Input places must have at least 1 token
        for idx in in_indices:
            tran_bdd &= manager.add_expr(curr_vars[idx])

        # Condition 2: Output places not in input must be empty
        for idx in out_indices:
            if idx not in in_indices:
                tran_bdd &= manager.add_expr(f"~{curr_vars[idx]}")

        # Condition 3: Update tokens
        for i in range(n):
            curr_var = curr_vars[i]
            next_var = next_vars[i]

            if i in in_indices and i not in out_indices:
                # Token consumed
                tran_bdd &= manager.add_expr(f"~{next_var}")
            elif i not in in_indices and i in out_indices:
                # Token produced
                tran_bdd &= manager.add_expr(next_var)
            elif i in in_indices and i in out_indices:
                # Token stays (self-loop)
                tran_bdd &= manager.add_expr(f"({curr_var} & {next_var}) | (~{curr_var} & ~{next_var})")
            else:
                # Place not involved in transition
                tran_bdd &= manager.add_expr(f"({curr_var} & {next_var}) | (~{curr_var} & ~{next_var})")

        T |= tran_bdd

    # Compute reachable states using fixed point iteration
    R_old = None
    while R_old != R:
        R_old = R

        # Image computation: ∃curr. (R & T)
        # We need to rename next_vars to curr_vars for the image
        image = manager.let({next_vars[i]: curr_vars[i] for i in range(n)},
                           manager.quantify(R & T, curr_vars, forall=False))

        R |= image

    # Count reachable states
    count = manager.count(R, nvars=n)

    memory_after = process.memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    running_time = time.time() - start_time

    if verbose:
        print(f"Reachable markings: {count}")
        markings = enumerate_bdd_markings(R, manager, curr_vars, places_list)
        for x in markings:
            print(x)
        print(f"Running time: {running_time:.6f} s")
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Memory used: {memory_used:.2f} MB")


    return R, count, manager, curr_vars




# task 4
# Helper function to enumerate markings from BDD
def enumerate_bdd_markings(R, manager, curr_vars, places_list):
    """Enumerate all markings from a BDD."""
    markings = []

    # Convert BDD to list of assignments
    assignments = manager.pick_iter(R, care_vars=curr_vars)

    for assign in assignments:
        marking = {}
        for i, place in enumerate(places_list):
            var = curr_vars[i]
            marking[place] = 1 if assign[var] else 0
        markings.append(marking)

    return markings


def detect_deadlock_bdd_ilp(net: PetriNet, verbose: bool = False , timeout_seconds: int = None):
    """
    Combine BDD (reachable set) and ILP:
      - Get R = compute_reachable_bdd(net)
      - Enumerate all reachable markings
      - Use ILP to select 1 reachable marking where no transition is enabled
    Returns (found: bool, marking: dict|None)
    """
    start_time = time.time()

    # Compute reachable set using BDD
    R, count, manager, curr_vars = bbd(net, False)
    if verbose:
        print(f"[BDD] satisfy_count = {count}")

    # Get places list
    places_list = sorted(net.places.keys())

    # Enumerate markings
    markings = enumerate_bdd_markings(R, manager, curr_vars, places_list)
    if verbose:
        print(f"[BDD] Enumerated {len(markings)} markings")
    if len(markings) == 0:
        return False, None

    # Build input/output structures for enable check
    inp = {}
    outp = {}
    inp_w = {}
    outp_w = {}
    for s, t, w in net.arcs:
        if s in net.places:
            inp.setdefault(t, []).append(s)
            inp_w[(s, t)] = w
        elif s in net.transitions:
            outp.setdefault(s, []).append(t)
            outp_w[(s, t)] = w

    # Precompute which transitions are enabled in each marking
    K = len(markings)
    e_tk = {t: [0] * K for t in net.transitions}

    for k, m in enumerate(markings):
        for t in net.transitions:
            tran_in = inp.get(t, [])
            tran_out = outp.get(t, [])

            # Check if transition is enabled
            fire = all(m[p] >= inp_w.get((p, t), 1) for p in tran_in)
            fire = fire and all(m[p] == 0 for p in tran_out if p not in tran_in)

            e_tk[t][k] = 1 if fire else 0

    # Build ILP problem
    prob = pulp.LpProblem("deadlock_detection", pulp.LpMinimize)
    prob += 0  # Dummy objective

    # Variables: z_k = 1 if marking k is selected
    z = [pulp.LpVariable(f"z_{k}", lowBound=0, upBound=1, cat='Binary') for k in range(K)]
    prob += pulp.lpSum(z) == 1  # Select exactly one marking

    # Variables: y_t = 1 if transition t is enabled in selected marking
    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0, upBound=1, cat='Binary') for t in net.transitions}

    # Constraints linking y_t and z_k
    for t in net.transitions:
        # y_t <= sum_k e_tk * z_k
        prob += y[t] <= pulp.lpSum(e_tk[t][k] * z[k] for k in range(K))

        # For each marking where t is enabled: y_t >= z_k
        for k in range(K):
            if e_tk[t][k] == 1:
                prob += y[t] >= z[k]

    # Deadlock constraint: no transition enabled
    prob += pulp.lpSum(y[t] for t in net.transitions) == 0

    # Solve
    if timeout_seconds:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout_seconds)
    else:
        solver = pulp.PULP_CBC_CMD(msg=0)

    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    running_time = time.time() - start_time

    if verbose:
        print(f"Running time: {running_time:.6f} s")

    if status in ("Optimal", "Feasible"):
        # Find the selected marking
        chosen = None
        for k in range(K):
            if pulp.value(z[k]) > 0.5:
                chosen = k
                break

        if chosen is not None:
            if verbose:
                print(f"Deadlock marking found: {markings[chosen]}")

            # clean bbd
            del manager
            return True, markings[chosen]

    if verbose:
        print("No deadlock found")
    # clean bbd
    del manager
    return False, None

# task 5

def optimize_reachable_marking(net: PetriNet, cost_list, verbose=False):
    start_time = time.time()

    places_list = sorted(net.places.keys())
    n = len(places_list)

    if len(cost_list) != n:
        raise ValueError(f"Cost list must have exactly {n} elements (one per place)")

    print("Cost :", cost_list)
    
    if verbose:
        print(f"[Binary Search] Places: {places_list}")
        print(f"[Binary Search] Costs: {cost_list}")

    # Compute reachable set using BDD
    R, total_count, manager, curr_vars = bbd(net, False)
    if verbose:
        print(f"[Binary Search] Total reachable states: {total_count}")

    # Binary search for optimal cost
    min_cost = 0
    max_cost = sum(cost_list)

    best_marking = None
    best_cost = -1

    # Helper function to check if there exists a marking with cost >= threshold
    def exists_marking_with_cost(threshold):
        # Build BDD for cost condition: sum(cost[i] * token[i]) >= threshold
        cost_bdd = build_cost_bdd(manager, curr_vars, cost_list, threshold)

        # Check intersection with reachable states
        intersection = R & cost_bdd
        return not intersection == manager.false

    # Binary search loop
    low, high = min_cost, max_cost
    while low <= high:
        mid = (low + high) // 2
        if verbose:
            print(f"[Binary Search] Testing cost >= {mid} (range [{low}, {high}])")

        if exists_marking_with_cost(mid):
            # Found marking with cost >= mid
            if verbose:
                print(f"  ✓ EXISTS marking with cost >= {mid}")
            best_cost = mid
            low = mid + 1

            # Get one such marking
            cost_bdd = build_cost_bdd(manager, curr_vars, cost_list, mid)
            intersection = R & cost_bdd
            assignment = manager.pick(intersection)

            if assignment:
                best_marking = {}
                for i, place in enumerate(places_list):
                    var = curr_vars[i]
                    best_marking[place] = 1 if assignment[var] else 0
        else:
            # No marking with cost >= mid
            if verbose:
                print(f"  ✗ NO marking with cost >= {mid}")
            high = mid - 1

    running_time = time.time() - start_time

    if verbose:
        print(f"[Binary Search] Optimal cost: {best_cost}")
        print(f"[Binary Search] Optimal marking: {best_marking}")
        print(f"[Binary Search] Running time: {running_time:.6f} s")

    return best_marking is not None, best_marking, best_cost

def build_cost_bdd(manager, curr_vars, cost_list, threshold):
    """Build BDD for condition: sum(cost[i] * token[i]) >= threshold"""
    n = len(curr_vars)

    # Use dynamic programming to build the BDD
    # dp[i][c] = BDD for achieving cost c with first i places

    # Initialize with false
    dp = {0: manager.true}  # cost 0 is always achievable (all places empty)

    for i in range(n):
        cost = cost_list[i]
        var = curr_vars[i]
        new_dp = {}

        for current_cost, bdd_expr in dp.items():
            # Case 1: place i is empty
            empty_cost = current_cost
            empty_bdd = bdd_expr & manager.add_expr(f"~{var}")

            if empty_cost in new_dp:
                new_dp[empty_cost] |= empty_bdd
            else:
                new_dp[empty_cost] = empty_bdd

            # Case 2: place i has token
            token_cost = current_cost + cost
            token_bdd = bdd_expr & manager.add_expr(var)

            if token_cost in new_dp:
                new_dp[token_cost] |= token_bdd
            else:
                new_dp[token_cost] = token_bdd

        dp = new_dp

    # Combine all costs >= threshold
    result = manager.false
    for cost, bdd_expr in dp.items():
        if cost >= threshold:
            result |= bdd_expr

    # clean bbd
    del manager
    return result




def run(file_name : str, cost : list()):
    net = read_pnmlFile("./file_test/" + file_name)
    print("Task 1:\n",net)


# task 2
    all_marking = all_reachable_marking(net)
    print("\n\n\nTask 2 : all_reachable_marking:")
    for x in all_marking:
        print(x)

# task 3
    print("\n\n\nTask 3:")
    R, count, manager, curr_vars = bbd(net, True) 
    places_list = sorted(net.places.keys())
    # Enumerate markings
    del manager

# task 4
    print("\n\n\nTask 4:")
    found, m = detect_deadlock_bdd_ilp(net, True)
# if found:
#     print("Deadlock detected !")

# task 5
    print("\n\nTask 5:")
    found, marking, opt_value = optimize_reachable_marking(net, cost, True)
