from typing import List, Set, Tuple, Dict, Union
from utils.ast_analyzer import ASTVarAnalyzer
from utils.ast_def import ASTNode, json2astNode
from utils.ast_traverse_util import ASTNodeASTProvider
import json

# symbolized variable names and function names
# however, here we don't actually symbolize them,
# we only collect the var name and func name to be symbolized
# and their corresponding symbolized name

symbolic_var_prefix = "VAR"
symbolic_func_prefix = "FUN"


def getVarFuncNamesInFunc(statements: List[ASTNode]) -> Tuple[Set[str], Set[str]]:
    astVarAnalyzer: ASTVarAnalyzer = ASTVarAnalyzer()
    varSets: Set[str] = set()
    funcSets: Set[str] = set()
    for statement in statements:
        provider: ASTNodeASTProvider = ASTNodeASTProvider()
        provider.node = statement
        astVarAnalyzer.analyzeAST(provider)
        varSets.update(astVarAnalyzer.variables)
        funcSets.update(astVarAnalyzer.funcNames)

    return varSets, funcSets


class SymbolizingTool(object):

    def __init__(self, systemDefinedVars: Set[str], func_names: Set[str]):
        self.systemDefinedVars: Set[str] = systemDefinedVars
        self.var2symbol: Dict[str, str] = dict()
        self.func2symbol: Dict[str, str] = dict()
        self.func_names = func_names

    # cpgs is all cpgs from functions of a program
    def getVarFuncNamesInFile(self, cpgs: List[Dict[str, Union[str, List[str]]]]):
        for cpg in cpgs:
            if cpg["functionName"] in self.func_names and cpg["functionName"] not in {"main"}:
                self.func2symbol[cpg["functionName"]] = symbolic_func_prefix + str(len(self.func2symbol) + 1)
            statements: List[ASTNode] = [json2astNode(json.loads(json_node)) for json_node in cpg["nodes"]]
            varSets, funcSets = getVarFuncNamesInFunc(statements)
            for var in varSets:
                if var not in self.var2symbol.keys() and var not in self.systemDefinedVars:
                    self.var2symbol[var] = symbolic_var_prefix + str(len(self.var2symbol) + 1)
            for func in funcSets:
                if func in self.func_names and func not in {"main"} and func not in self.func2symbol.keys():
                    self.func2symbol[func] = symbolic_func_prefix + str(len(self.func2symbol) + 1)


    def symbolize(self, code: str) -> str:
        tokens = code.split(' ')
        symbolized_tokens = []
        for token in tokens:
            symVarName: str = self.var2symbol.get(token, None)
            symFuncName: str = self.func2symbol.get(token, None)
            if symVarName is not None:
                symbolized_tokens.append(symVarName)
            elif symFuncName is not None:
                symbolized_tokens.append(symFuncName)
            else:
                symbolized_tokens.append(token)
        return " ".join(symbolized_tokens)