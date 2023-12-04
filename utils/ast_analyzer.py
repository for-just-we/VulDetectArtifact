from typing import List, Set, Dict
from utils.ast_traverse_util import ASTProvider, VariableEnvironment, VarDeclEnvironment, \
    MemberAccessVarEnvironment, CalleeEnvironment, CallVarEnvironment, \
    ClassStaticIdentifierVarEnvironment, IdentifierVarEnvironment
from utils.ast_def import ASTNode

class ASTVarAnalyzer(object):
    def __init__(self):
        self.environmentStack: List[VariableEnvironment] = list()
        self.variables: Set[str] = None
        self.var2type: Dict[str, str] = dict()
        self.funcNames: Set[str] = set()

    def reset(self):
        self.environmentStack.clear()
        self.variables = set()
        self.funcNames = set()
        # self.var2type = dict()

    def analyzeAST(self, astProvider: ASTProvider):
        self.reset()
        self.traverseAST(astProvider)


    def traverseAST(self, astProvider: ASTProvider):
        env: VariableEnvironment = self.createVarEnvironment(astProvider)
        self.traverseASTChildren(astProvider, env)


    def traverseASTChildren(self, astProvider: ASTProvider, env: VariableEnvironment):
        numChildren: int = astProvider.getChildCount()
        self.environmentStack.append(env)
        for i in range(numChildren):
            childProvider: ASTProvider = astProvider.getChild(i)
            self.traverseAST(childProvider)
        self.environmentStack.pop()
        self.variables.update(env.selfHandledSymbols())
        self.reportUpstream(env)
        self.var2type.update(env.var2type)
        self.funcNames.update(env.funcNames)


    def reportUpstream(self, env: VariableEnvironment):
        symbols: Set[str] = env.upstreamSymbols()
        astProvider: ASTProvider = env.astProvider
        if len(self.environmentStack) > 0:
            parentEnv: VariableEnvironment = self.environmentStack[-1]
            parentEnv.addChildSymbols(symbols, astProvider)



    def createVarEnvironment(self, astProvider: ASTProvider) -> VariableEnvironment:
        nodeType: str = astProvider.getTypeAsString()

        if nodeType == "IdentifierDecl" or nodeType == "Parameter":
            return VarDeclEnvironment(astProvider)
        elif nodeType == "CallExpression":
            return CallVarEnvironment(astProvider)
        elif nodeType == "ClassStaticIdentifier":
            return ClassStaticIdentifierVarEnvironment(astProvider)
        elif nodeType == "Identifier":
            return IdentifierVarEnvironment(astProvider)
        elif nodeType == "MemberAccess" or nodeType == "PtrMemberAccess":
            return MemberAccessVarEnvironment(astProvider)
        elif nodeType == "Callee":
            return CalleeEnvironment(astProvider)
        else:
            return VariableEnvironment(astProvider)

class SyVCChecker(object):
    def __init__(self, vul_funcs: Set[str]):
        self.vul_funcs: Set[str] = vul_funcs

    def visit(self, node: ASTNode):
        if node.type == "CallExpression":
            func_name = node.children[0].code
            if func_name in self.vul_funcs:
                return True
        elif node.type in {"ArrayIndexing", "PtrMemberAccess", "AdditiveExpression", "MultiplicativeExpression"}:
            return True

        flag = False
        for child in node.children:
            flag |= self.visit(child)
        return flag


if __name__ == '__main__':
    nodes = [
            "{\"line\":37,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"IdentifierDeclStatement\",\"charVoid structCharVoid ;\",\"\"],[\"IdentifierDecl\",\"structCharVoid\",\"\"],[\"IdentifierDeclType\",\"charVoid\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
            "{\"line\":38,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . voidSecond = ( void * ) SRC_STR ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . voidSecond = ( void * ) SRC_STR\",\"=\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( void * ) SRC_STR\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"],[\"CastTarget\",\"void *\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"]]}",
            "{\"line\":40,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}",
            "{\"line\":42,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[8,11],[8,12],[10,13],[10,14]],\"contents\":[[\"ExpressionStatement\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) ) ;\",\"\"],[\"CallExpression\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) )\",\"\"],[\"Callee\",\"memcpy\",\"\"],[\"ArgumentList\",\"structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"memcpy\",\"\"],[\"Argument\",\"structCharVoid . charFirst\",\"\"],[\"Argument\",\"SRC_STR\",\"\"],[\"Argument\",\"sizeof ( structCharVoid )\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
            "{\"line\":43,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[4,6],[4,7],[5,8],[5,9],[8,10],[8,11],[10,12],[10,13],[11,14],[11,15],[13,16],[13,17]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0'\",\"=\"],[\"ArrayIndexing\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ]\",\"\"],[\"CharExpression\",\"'\\\\0'\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"AdditiveExpression\",\"( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1\",\"-\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"MultiplicativeExpression\",\"sizeof ( structCharVoid . charFirst ) / sizeof ( char )\",\"/\"],[\"IntegerExpression\",\"1\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid . charFirst )\",\"\"],[\"SizeofExpr\",\"sizeof ( char )\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"SizeofOperand\",\"char\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
            "{\"line\":44,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . charFirst ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . charFirst )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
            "{\"line\":45,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}"
        ]

    import json
    from utils.ast_def import json2astNode
    from utils.ast_traverse_util import ASTNodeASTProvider
    json_nodes: List[dict] = [json.loads(n) for n in nodes]
    stmts: List[ASTNode] = [json2astNode(n) for n in json_nodes]