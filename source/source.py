from queue import Queue
import psutil
import os
import time
import xml.etree.ElementTree as et
from collections import defaultdict, namedtuple

# BDD
import pyeda.inter as pyeda
from pyeda.inter import exprvars,bddvars, expr, expr2bdd

# ILP
import pulp

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

def read_pnmlFile(filepath: str) -> PetriNet:
    tree = et.parse(filepath)
    root = tree.getroot()
    
    net = PetriNet()
    
    #   delete namespace 
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]  # remove namespace
    
    #   find all place 
    for place in root.findall(".//place"):
        pid = place.get("id")
        initial_marking = place.find("initialMarking")
        if initial_marking is not None:
            text_elem = initial_marking.find("text")
            if text_elem is not None and text_elem.text:
                net.places[pid] = text_elem.text
            else:
                net.places[pid] = "0"
        else:
            net.places[pid] = "0"
    
    #   find all transition 
    for tran in root.findall(".//transition"):
        tid = tran.get("id")
        name_elem = tran.find("name")
        if name_elem is not None:
            text_elem = name_elem.find("text")
            if text_elem is not None and text_elem.text:
                net.transitions[tid] = text_elem.text
            else:
                net.transitions[tid] = tid
        else:
            net.transitions[tid] = tid
    
    #   find all arc 
    for arc in root.findall(".//arc"):
        source = arc.get("source")
        target = arc.get("target")
        
        inscription = arc.find("inscription")
        if inscription is not None:
            text_elem = inscription.find("text")
            if text_elem is not None and text_elem.text:
                try:
                    weight = int(text_elem.text)
                except ValueError:
                    weight = 1
            else:
                weight = 1
        else:
            weight = 1
        
        net.arcs.append((source, target, weight))
    
    return net

# task 2


def all_reachable_marking(net : PetriNet)->list[dict]:
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

    ini = {p : int(count) for p, count in net.places.items()}
    visited = [ini]

    queue = Queue()
    queue.put(ini)

    while not queue.empty():
        marking  = queue.get()
        for tran in net.transitions:
            tran_in = inp.get(tran, [])
            tran_out = outp.get(tran, [])

            fire = all(marking[p] >= inp_w.get((p, tran), 1) for p in tran_in)
            fire = fire and all(marking[p] == 0 for p in tran_out if p not in tran_in)
            
            if fire:
                new_marking = dict(marking)
                for i in tran_in:
                    new_marking[i] -= inp_w[(i, tran)]
                for o in tran_out:
                    new_marking[o] += outp_w.get((tran, o), 1)
                if new_marking not in visited:
                    visited.append(new_marking)
                    queue.put(new_marking)

    return visited


# task 3


def bbd(net: PetriNet, verbose : bool = False):
    # time
    start_time = time.time()
    # memory 
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB


    places_id = {j: i for i, j in enumerate(sorted(net.places.keys()))}
    n = len(places_id)

    curr = pyeda.bddvars('curr', n)
    next = pyeda.bddvars('next', n)

    R = curr[0] | ~curr[0]
    
    for i in net.places:
        id = places_id[i]
        val = net.places[i]
        if int(val) == 1:
            R &= curr[id]
        else:
            R &= ~curr[id]

    inVar = {tran: [] for tran in net.transitions}
    outVar = {tran: [] for tran in net.transitions}

    for s, t, w in net.arcs:
        if s in net.places and t in net.transitions:
            inVar.setdefault(t, []).append(s)
        elif s in net.transitions and t in net.places:
            outVar.setdefault(s, []).append(t)

    T = curr[0] & ~curr[0]

    for i in net.transitions:
        T_temp = curr[0] | ~curr[0]
        inVar_id = {places_id[j] for j in inVar.get(i, [])}
        outVar_id = {places_id[j] for j in outVar.get(i, [])}

        for j in inVar_id:
            T_temp &= curr[j]

        for j in outVar_id:
            if j not in inVar_id:
                T_temp &= ~curr[j]

        for j in range(n):
            inInVar = j in inVar_id
            inOutVar = j in outVar_id

            if inInVar and not inOutVar:
                T_temp &= ~next[j]
            elif not inInVar and inOutVar:
                T_temp &= next[j]
            elif inInVar and inOutVar:
                T_temp &= next[j]
            elif not inInVar and not inOutVar:
                T_temp &= (next[j] & curr[j]) | (~next[j] & ~curr[j])

        T |= T_temp
    
    
    trans = {j: i for i, j in zip(curr, next)}

    while True:
        R_curr = R

        R_img = (R & T).smoothing(curr)

        R_next = R_img.compose(trans)
            
        R = R | R_next

        if R == R_curr:
            break
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB    
    memory_used = memory_after - memory_before
    running_time = time.time() - start_time
    if verbose:
        print(f"reachable markings : {R.satisfy_count()}")
        print(f"Running_time : {running_time:.6f} s")
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after : {memory_after:.2f} MB")
        print(f"Memory used  : {memory_used:.2f} MB")  # ⭐ QUAN TRỌNG NHẤT
    return R, R.satisfy_count()   



# task 4 


# ---------------------------
# NEW: detect deadlock combining BDD & ILP 
# ---------------------------
def _enumerate_bdd_markings(R, net):
    """
    Trả về danh sách marking (list[dict]) theo cùng thứ tự places sorted(net.places.keys()).
    Sử dụng R.satisfy_all() nếu có, fallback bằng satisfy_one + blocking clause nếu cần.
    """
    places_list = sorted(net.places.keys())
    n = len(places_list)
    curr_vars = pyeda.bddvars('curr', n)

    markings = []
    try:
        gen = R.satisfy_all()
        for ass in gen:
            mark = {}
            for i, v in enumerate(curr_vars):
                val = ass.get(v, 0)
                mark[places_list[i]] = 1 if val else 0
            markings.append(mark)
        return markings
    except Exception:
        R_copy = R
        while True:
            one = R_copy.satisfy_one()
            if one is None:
                break
            mark = {}
            for i, v in enumerate(curr_vars):
                val = one.get(v, 0)
                mark[places_list[i]] = 1 if val else 0
            markings.append(mark)
            block = None
            for i, v in enumerate(curr_vars):
                lit = v if mark[places_list[i]] == 1 else ~v
                block = lit if block is None else (block & lit)
            if block is None:
                break
            R_copy = R_copy & ~block
        return markings

# ----------------------------
# ILP + BDD deadlock detection
# - Build BDD reachable R and enumerate markings M = {m1,...,mK}
# - Build ILP with binary selection z_k for each marking
# - For each transition t, compute e_tk = 1 if marking k enables t (considering same enable rules as BDD)
# - Introduce y_t (binary) indicating enabled in selected marking
# - Constraints:
#     sum_k z_k == 1
#     for each t: y_t <= sum_k e_tk * z_k
#                for each k with e_tk==1: y_t >= z_k
#     deadlock constraint: sum_t y_t == 0
# - If feasible => found dead reachable marking
# ----------------------------
#

def detect_deadlock_bdd_ilp(net: PetriNet, timeout_seconds: int = None, verbose: bool = False):
    """
    Kết hợp BDD (tập reachable) và ILP:
      - Lấy R = bdd(net)
      - Liệt kê tất cả marking reachable (theo R)
      - Dùng ILP chọn 1 marking reachable sao cho không có transition nào enabled
    Trả về (found: bool, marking: dict|None)
    """
    start_time = time.time()

    R, count = bbd(net)
    if verbose:
        print(f"[BDD] satisfy_count = {count}")

    markings = _enumerate_bdd_markings(R, net)
    if verbose:
        print(f"[BDD] Enumerated {len(markings)} markings")

    if len(markings) == 0:
        return False, None

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

    K = len(markings)
    e_tk = {t: [0]*K for t in net.transitions}
    for k, m in enumerate(markings):
        for t in net.transitions:
            tran_in = inp.get(t, [])
            tran_out = outp.get(t, [])
            fire = all(m[p] >= inp_w.get((p, t), 1) for p in tran_in)
            fire = fire and all(m[p] == 0 for p in tran_out if p not in tran_in)
            e_tk[t][k] = 1 if fire else 0

    prob = pulp.LpProblem("deadlock_detection", pulp.LpMinimize)
    prob += 0 

    z = [pulp.LpVariable(f"z_{k}", lowBound=0, upBound=1, cat='Binary') for k in range(K)]
    prob += pulp.lpSum(z) == 1

    y = {t: pulp.LpVariable(f"y_{t}", lowBound=0, upBound=1, cat='Binary') for t in net.transitions}

    for t in net.transitions:
        prob += y[t] <= pulp.lpSum(e_tk[t][k] * z[k] for k in range(K))
        for k in range(K):
            if e_tk[t][k] == 1:
                prob += y[t] >= z[k]

    prob += pulp.lpSum(y[t] for t in net.transitions) == 0

    # Bỏ thông báo solver hoàn toàn
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout_seconds) if timeout_seconds else pulp.PULP_CBC_CMD(msg=0)
    res = prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    running_time = time.time() - start_time
    
    if status in ("Optimal", "Feasible"):
        chosen = None
        for k in range(K):
            val = pulp.value(z[k])
            if val is not None and val > 0.5:
                chosen = k
                break
        if chosen is None:
            vals = [pulp.value(var) for var in z]
            chosen = max(range(K), key=lambda i: 0 if vals[i] is None else vals[i])

        if verbose:
            print(f"Running time: {running_time:.6f} s")
            print(f"Deadlock marking: {markings[chosen]}")
        
        return True, markings[chosen]
    else:
        if verbose:
            print(f"Running time: {running_time:.6f} s")
            print(f"No deadlock found")
        return False, None

# task 5


def optimize_reachable_marking(net: PetriNet, cost_list, verbose=False):
    
    start_time = time.time()

    places = sorted(net.places.keys())
    n = len(places)
    
    if len(cost_list) != n:
        raise ValueError("Cost list phải đúng bằng số lượng places")
    
    if verbose:
        print(f"[Binary Search] Places: {places}")
        print(f"[Binary Search] Costs: {cost_list}")
    
    # 1. Tính min và max cost có thể
    min_possible_cost = 0
    max_possible_cost = sum(cost_list)  # Tất cả places đều có token
    
    # 2. Lấy BDD reachable
    R, total_count = bbd(net)
    if verbose:
        print(f"[Binary Search] Total reachable states: {total_count}")
    
    # 3. Binary search
    low, high = min_possible_cost, max_possible_cost
    best_marking = None
    best_cost = -1
    
    curr_vars = pyeda.bddvars('curr', n)
    
    while low <= high:
        mid = (low + high) // 2
        if verbose:
            print(f"[Binary Search] Testing cost >= {mid} (range [{low}, {high}])")
        
        # Tạo BDD condition: cost >= mid
        cost_condition = create_cost_condition_bdd(curr_vars, places, cost_list, mid)
        
        # Kiểm tra: có marking nào vừa reachable vừa thỏa cost >= mid?
        exists = (R & cost_condition).satisfy_count() > 0
        
        if exists:
            # Tồn tại marking thỏa điều kiện → tăng lower bound
            if verbose:
                print(f"  ✓ EXISTS marking with cost >= {mid}")
            best_cost = mid
            low = mid + 1
            
            # Lưu một marking thỏa điều kiện (dùng cho kết quả cuối)
            one_marking = (R & cost_condition).satisfy_one()
            if one_marking:
                best_marking = decode_marking(one_marking, curr_vars, places)
        else:
            # Không tồn tại → giảm upper bound
            if verbose:
                print(f"  ✗ NO marking with cost >= {mid}")
            high = mid - 1
    
    running_time = time.time()-start_time
    if verbose:
        print(f"[Binary Search] Optimal cost: {best_cost}")
        print(f"[Binary Search] Optimal marking: {best_marking}")
        print(f"[Binary Search] running_time: {running_time: .6f} s")
    
    return best_marking is not None, best_marking, best_cost

def create_cost_condition_bdd(curr_vars, places, cost_list, threshold):
    """
    Tạo BDD biểu diễn điều kiện: sum(cost[i] * token[i]) >= threshold
    """
    n = len(places)
    
    # Đệ quy/tách để tạo điều kiện (tránh tạo biểu thức quá lớn)
    def create_condition_for_cost(target, start_idx):
        if target <= 0:
            # Đã đạt target → luôn true
            return expr(True)
        if start_idx >= n:
            # Hết places mà chưa đạt target → false
            return expr(False)
        
        current_place_cost = cost_list[start_idx]
        current_var = curr_vars[start_idx]
        
        # Case 1: Place này có token (cost được cộng)
        with_token = current_var & create_condition_for_cost(
            target - current_place_cost, start_idx + 1
        )
        
        # Case 2: Place này không có token (cost không thay đổi)
        without_token = ~current_var & create_condition_for_cost(
            target, start_idx + 1
        )
        
        return with_token | without_token
    
    return create_condition_for_cost(threshold, 0)

def decode_marking(bdd_assignment, curr_vars, places):
    """Chuyển BDD assignment thành marking dictionary"""
    marking = {}
    for i, place in enumerate(places):
        val = bdd_assignment.get(curr_vars[i], 0)
        marking[place] = 1 if val else 0
    return marking






# task 1
net = read_pnmlFile("./file_test/medium_petri_net.pnml")
print("Task 1:\n",net)
# task 2 
all_marking = all_reachable_marking(net)
print("\n\n\nTask 2 : all_reachable_marking:")
for x in all_marking:
    print(x)

# task 3
print("\n\n\nTask 3:")
bdd_marking = bbd(net,True)


# print("Debug task 2 + 3:")
# a,b = test_bdd_vs_task2(net)
#
# print("debug task 4:")
# # Chạy test
# found = test_specific_deadlock_marking(net)

# task 4
print("\n\n\nTask 4:")
found, m = detect_deadlock_bdd_ilp(net, 30, True)
# if found:
#     print("Deadlock detected !")


print("\n\nTask 5:")
cost = [2, 3, 1, 4, 5, 2, 3, 1, 4, 5, 2, 3]  # phải đúng số lượng places

found, marking, opt_value = optimize_reachable_marking(net, cost, True)

