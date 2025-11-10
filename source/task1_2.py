import xml.etree.ElementTree as et
# from collections import deque
from queue import Queue



'''
class petriNet:
    + places = {place_id : current mark in place}
    + transitimns = {transition_id : transition_name}
    + arcs = {(source, target, weight)}
'''

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []

def read_pnmlFile(filepath : str)->PetriNet:
    tree = et.parse(filepath)
    root = tree.getroot()
    
    net = PetriNet()

    for place in root.findall(".//place"):
        id = place.get("id")
        net.places[id] = place.findtext(".//initialMarking/text", default="0")

    for tran in root.findall(".//transition"):
        id = tran.get("id")
        net.transitions[id] = tran.findtext(".//name/text", default=id)

    # đọc arc kèm trọng số
    for arc in root.findall(".//arc"):
        source = arc.get("source")
        target = arc.get("target")
        weight = int(arc.findtext(".//inscription/text", default="1"))  # nếu không có thì =1
        net.arcs.append((source, target, weight))
    return net




# find all_reachable_marking
#   + input : PetriNet
#   + ouput : list of dict {place : tokens}
def all_reachable_marking(net : PetriNet)->list[dict]:
    inp = {}
    outp = {}
    inp_w = {}   # trọng số của cung vào transition
    outp_w = {}  # trọng số của cung ra transition

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

            # kiểm tra có thể fire (mỗi input place đủ token theo trọng số)
            fire = all(marking[p] >= inp_w[(p, tran)] for p in tran_in)
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

# Test
try:
    net = read_pnmlFile("./test_file_pnml/test1.pnml")
    print("Read file success!")
    print("Places:", net.places)
    print("Transitions:", net.transitions)
    print("Arcs:", net.arcs)
    
    reachable_markings = all_reachable_marking(net)
    print(f"\nFound {len(reachable_markings)} reachable markings:")
    for i, marking in enumerate(reachable_markings):
        print(f"Marking {i+1}: {marking}")
        
except FileNotFoundError:
    print("Error: File test file not found!")
except Exception as e:
    print(f"Error: {e}")
