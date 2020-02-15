import inspect
import astor
import ast
import os


def mod2src(module):
    s = inspect.getsource(module)
    return s


def src2ast(src):
    return ast.parse(src)


def mod2ast(mod):
    return src2ast(mod2src(mod))


def ast2src(ast):
    return astor.code_gen.to_source(ast)


def syft2framework_string(string, framework):
    return string.replace("Syft", framework.capitalize()).replace("syft", framework.lower())


def write_output_to_file(output, target_folder, module_name):


    f = open(target_folder + module_name + ".py", "w+")
    f.write(output)
    f.close()

# the root directory of the project into which you want to deposit the results
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("PySyft")[0] + "PySyft/"
