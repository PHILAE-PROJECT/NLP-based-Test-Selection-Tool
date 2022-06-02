# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:14:13 2020

@author: aahmad
"""

import csv
from pathlib import Path
from datetime import  datetime
import os
import sys
import agilkia


#  !!!!!! UPDATE THIS DEFAULT DIRECTORY VARIABLE !!!!!!!!
#  (or call this script with an explict directory name or an explicit  *.csv path)
csvTracesDirectory = r'scanette-master-traces'


def read_traces_csv(path: Path) -> agilkia.TraceSet:
    with path.open("r") as input:
        trace1 = agilkia.Trace([])
        for line in csv.reader(input):
            timestr = line[1].strip()
            timestamp = datetime.fromtimestamp(int(timestr) / 1000.0)
            sessionID = line[2].strip()
            objInstance = line[3].strip()
            action = line[4].strip()
            paramstr = line[5].strip()
            result = line[6].strip()
            if paramstr == "[]":
                inputs = {}
            else:
                if  paramstr.startswith("[") and paramstr.endswith("]"):
                    paramstr = paramstr[1:-1]
                inputs = {"param" : paramstr}
            if( result == "?" or result =="\"?\""):
                outputs = {}
            else:
                outputs = {'Status': float(result)}
            others = {
                    'timestamp': timestamp,
                    'sessionID': sessionID,
                    'object': objInstance
                    }
            event = agilkia.Event(action, inputs, outputs, others)
            trace1.append(event)
    traceset = agilkia.TraceSet([])
    traceset.append(trace1)
    return traceset


def convert_csv_to_json(input_csv:Path) -> None:
    """Convert a single *.csv file into *.AgilkiaTraces.json file."""
    traceset = read_traces_csv(input_csv)
    traceset.set_event_chars({"scanner": ".", "abandon": "a", "supprimer": "-", "ajouter": "+",
                      "debloquer": "d", "fermerSession": "f", "transmission": "t", "payer":"p",
                      "ouvrirSession":"o"})
    traceset2 = traceset.with_traces_grouped_by("sessionID", property=True)
    data = traceset2.get_trace_data(method="action_counts")
    output = input_csv.with_suffix(".AgilkiaTraces.json")
    print(f"{input_csv} --> {output}")
    print(data.sum().sort_values())
    traceset2.save_to_json(output)


def main(path:Path) -> None:
    """Convert path (a *.csv file, or a directory of them) into Agilkia *.json files."""
    if path.exists() and path.suffix == ".csv":
        convert_csv_to_json(path)
    elif path.exists() and path.is_dir():
        for csv in path.glob("*.csv"):
            convert_csv_to_json(csv)
    else:
        print(f"ERROR: could not find {path}")


if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Arguments: traces.csv  (to convert one file to *.json)")
        print("Arguments: directory   (to convert all *.csv files to *.json)")
        print(f"          (trying default directory: {csvTracesDirectory})")
        main(Path(csvTracesDirectory))
    else:
        for path in sys.argv[1:]:
            main(Path(path))