from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import numpy as np
import javalang


def parse_java_code(code):
    tree = javalang.parse.parse(code)  # 生成 AST
    return tree


def ast_to_sequence(ast):
    sequence = []
    for _, node in ast:
        if isinstance(node, javalang.tree.MethodInvocation):
            sequence.append(f"MethodInvocation:{node.member}")
        elif isinstance(node, javalang.tree.Statement):
            sequence.append(f"Statement:{type(node).__name__}")
    # 可以根据需要提取更多信息
    return sequence


def cosine_similarity_score(embedding1, embedding2):
    """
    计算余弦相似度(语义)。
    """
    return cosine_similarity([embedding1.numpy()], [embedding2.numpy()])[0][0]


def lexical_similarity(code1, code2):
    """
    词法相似度，使用Jaccard。
    """
    tokens1 = set(code1.split())
    tokens2 = set(code2.split())
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union != 0 else 0


def syntactic_similarity(code1, code2):
    try:
        """
        句法相似度【syntactic_similarity计算句法相似度】
        """
        ast1 = parse_java_code(code1)
        ast2 = parse_java_code(code2)

        seq1 = ast_to_sequence(ast1)
        seq2 = ast_to_sequence(ast2)

        distance = Levenshtein.distance(" ".join(seq1), " ".join(seq2))
        max_len = max(len(seq1), len(seq2))
        return 1 - (distance / max_len)
    except Exception as e:
        return 0


def combined_similarity(input_code, candidate_code, input_embedding, candidate_embedding):
    """
    计算三种相似度的平均值，并返回每种相似度。
    """
    cosine_sim = cosine_similarity_score(input_embedding, candidate_embedding)
    lexical_sim = lexical_similarity(input_code, candidate_code)
    syntactic_sim = syntactic_similarity(input_code, candidate_code)
    average_similarity = np.mean([cosine_sim, lexical_sim, syntactic_sim])
    return average_similarity, cosine_sim, lexical_sim, syntactic_sim