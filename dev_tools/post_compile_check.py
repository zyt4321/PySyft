from syft.generic.compiler.util import ROOT_DIR
from os import listdir
from os.path import isfile, join
import logging


def get_code_contents_for_framework(framework="torch"):
    d = ROOT_DIR + f"syft/_{framework}/tensor"
    onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]

    name2contents = {}

    for file_name in onlyfiles:
        f = open(d + "/" + file_name, 'r')
        contents = f.readlines()
        f.close()

        contents.append("<<<<END OF FILE>>>>")

        name2contents[d + "/" + file_name] = contents
    return name2contents


def check_for_illegal_strings(name2contents, illegal_strings):
    for name, contents in name2contents.items():
        for _line_i, line in enumerate(contents):
            line_i = _line_i + 1  # enumeration is off by one for how most IDEs index
            for bad_str, explanation, recommendation in illegal_strings:
                if bad_str in line and "#IGNORE_IS_" + bad_str not in line:
                    msg = f"Found illegal string '{bad_str}' on line {line_i} of file {name}"
                    msg += f"\n\n\t{line_i - 1}\t{contents[_line_i - 1]}"
                    msg += f">>>\t{line_i}\t{contents[_line_i]}"
                    msg += f"\t{line_i + 1}\t{contents[_line_i + 1]}"
                    msg += f"\n\tREASON FOR WARNING:{explanation}\n"
                    msg += f"\n\tRECOMMENDED FIX:{recommendation}\n\n"
                    logging.warning(msg)


def check_framework(framework, resources):
    name2contents = get_code_contents_for_framework(framework.lower())
    check_for_illegal_strings(name2contents, resources['illegal_strings'])


### BEGIN CONFIGURATION

### PyTorch Automatic Compilation Configuration

torch_resources = {}

illegal_strings = list()
illegal_strings.append(("Syft",
                        "A capitalized version of the word 'Syft' often refers to a class which was"
                        "\n\t\t not properly code generated but was simply copied from the source file."
                        , "You will probably need to modify an existing Transformer or add a new one to ensure"
                          "\n\t\t that this abtract tensor gets properly generated. You can hide this warning by adding a comment on"
                          "\n\t\t the line in question: #IGNORE_IS_Syft )"))

torch_resources['illegal_strings'] = illegal_strings

### Numpy Automatic Compilation Configuration
numpy_resources = {}

illegal_strings = list()

numpy_resources['illegal_strings'] = illegal_strings

### END CONFIGURATION


check_framework("torch", torch_resources)