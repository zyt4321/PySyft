from importchecker.importchecker import ast, findModules, os, ImportDatabase, Module
import sys
import tempfile

def cleanup(output):

    new_output = ""

    for line in output.split("\n"):
        if("import" in line and "_only" in line):
            ""
        else:
            new_output += line + "\n"

    return new_output


def remove_unused_imports(output):
    file = tempfile.NamedTemporaryFile(mode='w+t')
    file.file.write(output)
    file.file.flush()

    lines_to_remove = unused_import_line_numbers(file.name)

    new_output = ""

    for _line_i, line in enumerate(output.split("\n")):
        line_i = _line_i + 1
        if (line_i in lines_to_remove):
            ""
        else:
            new_output += line + "\n"

    file.close()
    return new_output

def unused_import_line_numbers(path, cwd=None, stdout=None):
    line_numbers = list()

    cwd = cwd or os.getcwd()
    lencwd = len(cwd) + 1

    try:
        path = path or sys.argv[1]
    except IndexError:
        print(u"No path supplied", file=stdout)
        sys.exit(1)

    fullpath = os.path.abspath(path)
    path = fullpath
    if not os.path.isdir(fullpath):
        path = os.path.dirname(fullpath)

    db = ImportDatabase(path)
    if os.path.isdir(fullpath):
        db.findModules()
    else:
        db.addModule(Module(fullpath))

    unused_imports = db.getUnusedImports()
    module_paths = sorted(unused_imports.keys())
    for path in module_paths:
        info = unused_imports[path]
        if path.startswith(cwd):
            path = path[lencwd:]
        if not info:
            continue
        line2names = {}
        for name, line in info:
            names = line2names.get(line, [])
            names.append(name)
            line2names[line] = names
        lines = sorted(line2names.keys())
        for line in lines:
            names = ', '.join(line2names[line])
            #             print(u"{}:{}: {}".format(path, line, names), file=stdout)
            line_numbers.append(line)
    return line_numbers

