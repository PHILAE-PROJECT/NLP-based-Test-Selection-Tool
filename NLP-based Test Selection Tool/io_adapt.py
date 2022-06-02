import csvToAgilkia
import ast
from configurations import input_file
import os
import agilkia
from pathlib import Path


def make_output_dir(input_name):
    dir_name = Path(input_name).stem + "_out"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        print("Using already existing directory : ", dir_name)


def csv_to_json(input_name):
    csvToAgilkia.convert_csv_to_json(Path(input_name))


def is_json(input_name):
    if input_name.endswith(".json"):
        return True
    return False


def get_outputs_names(input_name):
    if not is_json(input_name):
        json_in_name = Path(input_name).stem + ".AgilkiaTraces.json"
    else:
        json_in_name = input_name
    stem = Path(input_name).stem
    json_out_name = stem + "_out.json"
    csv_out_name = stem + "_out.json"
    dir_name = stem + "_out/"
    return json_in_name, json_out_name, csv_out_name, dir_name


def prepare_input():
    make_output_dir(input_file)
    if not is_json(input_file):
        csv_to_json(input_file)


def save_output_json(traces, cluster_info, input_name):
    add_cluster_info(traces, cluster_info)


def make_output_json(input_name, output_name, cluster_info):
    traces = agilkia.TraceSet.load_from_json(
        Path(get_outputs_names(input_name)[0]))
    add_cluster_info(traces, cluster_info)
    dir_name = Path(input_name).stem + "_out/"
    file_name = Path(output_name).stem + ".json"
    traces.save_to_json(Path(dir_name + file_name))


def add_cluster_info(traces, cluster_info):
    labels = [(-1) for _ in traces.traces]
    for i, j in cluster_info:
        labels[i] = j
    traces.set_clusters(labels)
