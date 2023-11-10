from typing import List

class ASTNode:
    def __init__(self, type: str, code: str):
        self.type: str = type
        self.code: str = code
        self.children: List[ASTNode] = list()
        self.childNumber = -1

'''
an example is: {   "line":37,
                    "edges":[[0,1],[1,2],[1,3]],
                    "contents":[
                                    ["IdentifierDeclStatement","charVoid structCharVoid ;"],
                                    ["IdentifierDecl","structCharVoid"],
                                    ["IdentifierDeclType","charVoid"],
                                    ["Identifier","structCharVoid"]]
                                    }
'''
def json2astNode(data: dict) -> ASTNode:
    types = list(map(lambda content: content[0], data["contents"]))
    tokenSeqs = list(map(lambda content: content[1], data["contents"]))
    nodes: List[ASTNode] = [ASTNode(type, tokenSeq) for type, tokenSeq in zip(types, tokenSeqs)]
    for edge in data["edges"]:
        nodes[edge[1]].childNumber = len(nodes[edge[0]].children)
        nodes[edge[0]].children.append(nodes[edge[1]])
    return nodes[0]




if __name__ == '__main__':
    sample = {   "line":37,
                    "edges":[[0,1],[1,2],[1,3]],
                    "contents":[
                                    ["IdentifierDeclStatement","charVoid structCharVoid ;"],
                                    ["IdentifierDecl","structCharVoid"],
                                    ["IdentifierDeclType","charVoid"],
                                    ["Identifier","structCharVoid"]]
                                    }
    node = json2astNode(sample)