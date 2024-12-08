import enum


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"   # 常规分类任务
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"  # seq2seq任务
    CAUSAL_LM = "CAUSAL_LM"  # LM任务
    TOKEN_CLS = "TOKEN_CLS"  # token的分类任务：序列标注之类的



