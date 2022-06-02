'''
 Created on : 01.06.2020
 @author    : Bahareh Afshinpour
 Last modify: 02.06.2022
 ---------------------------
'''
import numpy as np
from treelib import Node, Tree
import json
import configurations
import io_adapt
Action = []
InputParameters = []
Output = []
NameTypeOfOutput = []
x = []  # Type of Input parameters
y = []  # Type of result
Price = []
TypeofInputparameter = 0
NameofInputParameters = []
Barcode = []
Error = []
VectorOfTraces = []
CheckFlag = 0
Sessions = dict()


def is_float(s):
    result = False
    if s.count(".") == 1:
        if s.replace(".", "").isdigit():
            result = True
    return result


def string_format(i):
    if i.is_integer():
        return str(int(i))
    return str(i)


def append_event(event, configurations):
    tmp = []
    session_id = " " + event["meta_data"]["sessionID"]
    for col in Input_field_selection:
        if col in ["object", "timestamp", "sessionID"]:
            tmp.append(event["meta_data"][col])
        elif col == "action":
            tmp.append(event["action"])
        elif col == "input":
            if "param" in event["inputs"].keys():
                tmp.append("[" + event["inputs"]["param"] + "]")
            else:
                tmp.append("[]")
        elif col == "output":
            if "Status" in event["outputs"].keys():
                tmp.append(string_format(event["outputs"]["Status"]))
            else:
                tmp.append("?")
    Sessions.setdefault(session_id, []).append(tmp)


def create_tree(Sessions):
    tree = Tree()
    tree.create_node("Unlock", "-")  # root node

    for i in Sessions:
        id = "-"
        str1 = ""
        checkfirstnode = 0
        NewPath = Sessions[i]
        c = 0
        for j in NewPath:

            for k in j:
                str1 = str1+str(k)

            node = tree.get_node(id+str1)
            if(node == None):
                tree.create_node(tag=str(j), identifier=id +
                                 str1, parent=id, data=str(j))
            id = id+str1
        tree.create_node(tag=str(i), identifier=str(i), parent=id, data=str(i))

    tree.show()
    tree.save2file('tree_sessions.txt')


Input_field_selection = []
Input_field_selection = configurations.Input_field_selection
io_adapt.prepare_input()

print(Input_field_selection)
json_in_name = io_adapt.get_outputs_names(configurations.input_file)[0]

with open(json_in_name, 'r') as json_file:
    json_reader = json.load(json_file)
    for session in json_reader["traces"]:
        for event in session["events"]:
            append_event(event, configurations)


np.save("Sessions.npy", Sessions)
