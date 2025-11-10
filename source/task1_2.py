import xml.etree.ElementTree as et
from queue import Queue

class PetriNet:
    def __init__(self):
        self.places = {}
        self.transitions = {}
        self.arcs = []

def read_pnmlFile(filepath: str) -> PetriNet:
    tree = et.parse(filepath)
    root = tree.getroot()
    
    net = PetriNet()

    for place in root.findall(".//place"):
        id = place.get("id")
        initial_marking = place.findtext(".//initialMarking/text", default="0")
        net.places[id] = int(initial_marking)

    for tran in root.findall(".//transition"):
        id = tran.get("id")
        net.transitions[id] = tran.findtext(".//name/text", default=id)

    for arc in root.findall(".//arc"):
        source = arc.get("source")
        target = arc.get("target")
        weight = int(arc.findtext(".//inscription/text", default="1"))
        net.arcs.append((source, target, weight))
    
    return net

def all_reachable_marking(net: PetriNet) -> list[dict]:
    # Cấu trúc dữ liệu để lưu thông tin về các cung
    inp = {}  # place -> transition
    outp = {}  # transition -> place  
    inp_w = {}  # trọng số cung input: (place, transition) -> weight
    outp_w = {}  # trọng số cung output: (transition, place) -> weight

    # Phân loại các cung và lưu trọng số
    for source, target, weight in net.arcs:
        if source in net.places:  # Cung từ place đến transition
            inp.setdefault(target, []).append(source)
            inp_w[(source, target)] = weight
        elif source in net.transitions:  # Cung từ transition đến place
            outp.setdefault(source, []).append(target)
            outp_w[(source, target)] = weight

    # Marking ban đầu
    initial_marking = {p: count for p, count in net.places.items()}
    
    visited = {tuple(sorted(initial_marking.items()))}
    queue = Queue()
    queue.put(initial_marking)

    while not queue.empty():
        marking = queue.get()
        
        for transition_id in net.transitions:
            # Kiểm tra transition có thể bắn
            transition_inputs = inp.get(transition_id, [])
            
            can_fire = True
            for place_id in transition_inputs:
                required_tokens = inp_w.get((place_id, transition_id), 1)
                if marking.get(place_id, 0) < required_tokens:
                    can_fire = False
                    break
            
            if can_fire:
                new_marking = dict(marking)
                
                for place_id in transition_inputs:
                    required_tokens = inp_w.get((place_id, transition_id), 1)
                    new_marking[place_id] -= required_tokens
                
                transition_outputs = outp.get(transition_id, [])
                for place_id in transition_outputs:
                    added_tokens = outp_w.get((transition_id, place_id), 1)
                    new_marking[place_id] = new_marking.get(place_id, 0) + added_tokens
                
                new_tuple = tuple(sorted(new_marking.items()))
                if new_tuple not in visited:
                    visited.add(new_tuple)
                    queue.put(new_marking)

    return [dict(v) for v in visited]

# Test
try:
    net = read_pnmlFile("test1.pnml")
    print("Read file success!")
    print("Places:", net.places)
    print("Transitions:", net.transitions)
    print("Arcs:", net.arcs)
    
    reachable_markings = all_reachable_marking(net)
    print(f"\nFound {len(reachable_markings)} reachable markings:")
    for i, marking in enumerate(reachable_markings):
        print(f"Marking {i+1}: {marking}")
        
except FileNotFoundError:
    print("Error: File 'test2.pnml' not found!")
except Exception as e:
    print(f"Error: {e}")
