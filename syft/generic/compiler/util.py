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
    return string.replace("Syft", framework.capitalize() ).replace("syft", framework.lower())

def get_complier_resources(base_module, module_name, framework):
    # folder to deposit each Torch tensor
    target_folder = ROOT_DIR + f"syft/_{framework.lower()}/tensor/"

    print(f"Generic ({module_name}.py) -> {framework.capitalize()} ({target_folder.split('PySyft')[1][1:]}{module_name}.py)")

    module = getattr(base_module, module_name)

    tree = mod2ast(module)

    return tree, target_folder

def write_ast_to_file(tree, target_folder, module_name):

    output = ast2src(tree)
    f = open(target_folder + module_name + ".py", 'w+')
    f.write(output)
    f.close()

# the root directory of the project into which you want to deposit the results
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("PySyft")[0] + "PySyft/"