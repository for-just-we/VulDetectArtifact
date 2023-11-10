from typing import Set, Dict
from utils.ast_def import ASTNode
from abc import abstractmethod

class ASTProvider(object):
    @abstractmethod
    def getTypeAsString(self) -> str:
        pass

    @abstractmethod
    def getChild(self, i: int):
        pass

    @abstractmethod
    def getEscapedCodeStr(self) -> str:
        pass

    @abstractmethod
    def getChildNumber(self) -> int:
        pass

    @abstractmethod
    def getChildCount(self) -> int:
        pass


class ASTNodeASTProvider(ASTProvider):
    def __init__(self):
        self.node: ASTNode = None

    def getTypeAsString(self) -> str:
        return self.node.type

    def getChild(self, i: int) -> ASTProvider:
        childProvider: ASTNodeASTProvider = ASTNodeASTProvider()
        childProvider.node = self.node.children[i]
        return childProvider

    def getChildCount(self) -> int:
        return len(self.node.children)

    def getEscapedCodeStr(self) -> str:
        return self.node.code

    def getChildNumber(self) -> int:
        return self.node.childNumber

    def __eq__(self, other):
        if not isinstance(other, ASTNodeASTProvider):
            return False
        return self.node == other.node

class VariableEnvironment(object):
    def __init__(self, astProvider: ASTProvider):
        self.astProvider: ASTProvider = astProvider
        # 交由父节点处理的token
        self.handledSymbols: Set[str] = set()
        # 交由父节点处理的token
        self.upStreamSymbols: Set[str] = set()
        self.var2type: Dict[str, str] = dict() # 变量名映射为类型名
        self.funcNames: Set[str] = set() # 使用过的函数名

    # 处理子结点的symbol，默认自己解决子结点中的symbol
    def addChildSymbols(self, childSymbols: Set[str], child: ASTProvider):
        self.handledSymbols.update(childSymbols)

    # 交由父节点处理的symbol
    def upstreamSymbols(self) -> Set[str]:
        return self.upStreamSymbols

    # 自己处理的symbol
    def selfHandledSymbols(self) -> Set[str]:
        return self.handledSymbols

# 函数调用环境
class CallVarEnvironment(VariableEnvironment):
    def addChildSymbols(self, childSymbols: Set[str], child: ASTProvider):
        childNumber: int = child.getChildNumber()
        # 函数名不添加
        if childNumber != 0:
            # 参数中的变量名全都处理了
            self.handledSymbols.update(childSymbols)

    # 交由父节点处理的symbol
    def upstreamSymbols(self) -> Set[str]:
        return set()

    # 自己处理的symbol
    def selfHandledSymbols(self) -> Set[str]:
        return set()

class CalleeEnvironment(VariableEnvironment):
    def addChildSymbols(self, childSymbols: Set[str], child: ASTProvider):
        self.funcNames.update(childSymbols)


class ClassStaticIdentifierVarEnvironment(VariableEnvironment):
    # Identifier类型直接获取token作为symbol，并返回给父节点处理
    def upstreamSymbols(self) -> Set[str]:
        code: str = self.astProvider.getChild(1).getEscapedCodeStr()
        retval: Set[str] = { code }
        return retval

    def selfHandledSymbols(self) -> Set[str]:
        return set()

class IdentifierVarEnvironment(VariableEnvironment):
    # Identifier类型直接获取token作为symbol，并返回给父节点处理
    def upstreamSymbols(self) -> Set[str]:
        code: str = self.astProvider.getEscapedCodeStr()
        retval: Set[str] = { code }
        return retval

    def selfHandledSymbols(self) -> Set[str]:
        return set()

# 结构体成员访问
# 这里需要注意的是会出现 struct1 -> inner1 这种，我会将struct1 和 struct1 -> inner1 添加到变量列表，但是inner1就不会了
class MemberAccessVarEnvironment(VariableEnvironment):
    def upstreamSymbols(self) -> Set[str]:
        retval: Set[str] = { self.astProvider.getEscapedCodeStr() }
        return retval

    def addChildSymbols(self, childSymbols: Set[str], child: ASTProvider):
        # 结构体变量名添加到symbol中但是使用的成员变量名不添加
        childNum: int = child.getChildNumber()
        if childNum == 0:
            self.handledSymbols.update(childSymbols)

# 变量定义
class VarDeclEnvironment(VariableEnvironment):
    def __init__(self, astProvider: ASTProvider):
        super().__init__(astProvider)
        self.type: str = self.astProvider.getChild(0).getEscapedCodeStr()

    def addChildSymbols(self, childSymbols: Set[str], child: ASTProvider):
        # 结构体变量名添加到symbol中但是使用的成员变量名不添加
        childNum: int = child.getChildNumber()
        # 变量名
        if childNum == 1:
            for symbol in childSymbols:
                self.var2type[symbol] = self.type
            self.handledSymbols.update(childSymbols)