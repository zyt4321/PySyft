import syft as sy
import torch as th
import random


def create_zk_resources(dummy_data, dummy_model, plan):
    @sy.workers.utils.func2plan(dummy_data, dummy_model)
    def forward_prop(dummy_data, dummy_model):
        return plan(dummy_data, dummy_model)

    _data_ids = [dummy_data.id]
    _model_ids = [dummy_model.id]

    graph = forward_prop.plan_graph

    id2tensor = {}
    id2tensor[dummy_data.id] = dummy_data
    id2tensor[dummy_model.id] = dummy_model

    return _data_ids, _model_ids, graph, id2tensor


class ZKCircuit:
    def __init__(self, dummy_data, dummy_model, forward_prop, verbose=True):

        pdids, pmids, graph, _ = create_zk_resources(dummy_data, dummy_model, forward_prop)

        self.circuit_id = random.randint(0, 1e10)
        self.plan_data_ids = pdids
        self.plan_model_ids = pmids
        self.graph = graph
        self.graph_file_path = "zk_graph_" + str(self.circuit_id) + ".txt"
        self.input_file_path = "zk_input_" + str(self.circuit_id) + ".csv"
        self.forward_prop = forward_prop

        self.dummy_model = dummy_model
        self.verbose = verbose

        self.create_graph_file()

        if self.verbose:
            print("Updated file...", self.graph_file_path)

    def create_graph_file(self):
        f = open(self.graph_file_path, "w")
        lines = list()
        for row in self.graph:
            line = ""
            for entry in row:

                if isinstance(entry, bytes):
                    entry = entry.decode("ascii")

                line += str(entry) + ","

            line = line[:-1] + "\n"
            lines.append(line)
        f.writelines(lines)
        f.close()

    def create_pred_file(self, real_data):

        real_data_ids, real_model_ids, _, real_id2tensor = create_zk_resources(
            real_data, self.dummy_model, self.forward_prop
        )

        out_list = ["index,is_input,is_public,value\n"]
        for real_data_id, plan_data_id in list(zip(real_data_ids, self.plan_data_ids)):
            out_list.append(
                str(plan_data_id) + ",1,1," + str(int(real_id2tensor[real_data_id][0])) + "\n"
            )

        for real_model_id, plan_model_id in list(zip(real_model_ids, self.plan_model_ids)):
            out_list.append(
                str(plan_model_id) + ",0,1," + str(int(real_id2tensor[real_model_id][0])) + "\n"
            )

        f = open(self.input_file_path, "w")
        f.writelines(out_list)
        f.close()

    def predict(self, input):
        circuit.create_pred_file(real_data=input)

        if self.verbose:
            print("Updated file...", self.input_file_path)

        # magic code which runs prediction on command line

        # magic code which gets result


dummy_data = th.tensor([1])
dummy_model = th.tensor([1])


def forward_prop(data, model):
    return data * model + data * model


circuit = ZKCircuit(dummy_data, dummy_model, forward_prop)
