from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score


class MessageType(Enum):
    LEGIT = 1
    SPMSG = 2


@dataclass
class Message:
    msg_type: MessageType
    header: list[int]
    body: list[int]


def read_all_messages() -> list[Message]:
    data = []

    for dirpath, _, filenames in os.walk(Path("messages")):
        for filename in filenames:
            with open(Path(dirpath) / filename) as message:
                if "legit" in filename:
                    msg_type = MessageType.LEGIT
                else:
                    msg_type = MessageType.SPMSG

                for line in message.readlines():
                    line = line.rstrip("\n")
                    if line.startswith("Subject"):
                        header = list(map(int, filter(lambda s: len(s) > 0, line.split(" ")[1:])))
                    elif len(line) == 0:
                        continue
                    else:
                        body = list(map(int, filter(lambda s: len(s) > 0, line.split(" "))))

                data.append(Message(msg_type, header, body))

    return data


def extract_message_types(data: list[Message]) -> list[int]:
    return list(map(lambda m: m.msg_type.value - 1, data))


def split_train_test(
    vectorized: csr_matrix,
    msg_types: list[int],
    test_prop: float,
) -> tuple[csr_matrix, list[int], csr_matrix, list[int]]:
    data = vectorized.toarray()
    legit = []
    spmsg = []

    for i in range(len(data)):
        if msg_types[i] == 0:
            legit.append(data[i])
        else:
            spmsg.append(data[i])

    legit_train_size = round(len(legit) * (1.0 - test_prop))
    spmsg_train_size = round(len(spmsg) * (1.0 - test_prop))

    legit_train = legit[:legit_train_size]
    spmsg_train = spmsg[:spmsg_train_size]
    msg_types_train = [0] * legit_train_size + [1] * spmsg_train_size

    legit_test = legit[legit_train_size:]
    spmsg_test = spmsg[spmsg_train_size:]
    msg_types_test = [0] * (len(legit) - legit_train_size) + [1] * (len(spmsg) - spmsg_train_size)

    return (
        csr_matrix(legit_train + spmsg_train),
        msg_types_train,
        csr_matrix(legit_test + spmsg_test),
        msg_types_test,
    )


def display_roc_curve(title: str, y_true: list[int], y_score: np.ndarray) -> None:
    RocCurveDisplay.from_predictions(y_true, y_score[:, 1])
    plt.plot(np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2), linestyle="--")
    plt.title(title + f"\nAUC: {roc_auc_score(y_true, y_score[:, 1])}")
    plt.grid(True)
    plt.show()


def false_positive_count(y_true: list[int], y_pred: list[int]) -> int:
    count = 0

    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            count += 1

    return count


def one_hot_together(data: list[Message]) -> csr_matrix:
    n = len(data)
    m = 1 + max(map(lambda m: max(max(m.header, default=0), max(m.body, default=0)), data))
    one_hot_matrix = np.zeros((n, m))

    for i in range(n):
        for j in data[i].header:
            one_hot_matrix[i][j] = min(one_hot_matrix[i][j] + 1, 1)

        for j in data[i].body:
            one_hot_matrix[i][j] = min(one_hot_matrix[i][j] + 1, 1)

    return csr_matrix(one_hot_matrix)


def one_hot_separately(data: list[Message]) -> csr_matrix:
    n = len(data)
    m = 1 + max(map(lambda m: max(max(m.header, default=0), max(m.body, default=0)), data))
    one_hot_matrix = np.zeros((n, 2 * m))

    for i in range(n):
        for j in data[i].header:
            one_hot_matrix[i][j] = min(one_hot_matrix[i][j] + 1, 1)

        for j in data[i].body:
            one_hot_matrix[i][m + j] = min(one_hot_matrix[i][m + j] + 1, 1)

    return csr_matrix(one_hot_matrix)


def freq_together(data: list[Message]) -> csr_matrix:
    n = len(data)
    m = 1 + max(map(lambda m: max(max(m.header, default=0), max(m.body, default=0)), data))
    freq_matrix = np.zeros((n, m))

    for i in range(n):
        for j in data[i].header:
            freq_matrix[i][j] += 1

        for j in data[i].body:
            freq_matrix[i][j] += 1

    return csr_matrix(freq_matrix)


def freq_separately(data: list[Message]) -> csr_matrix:
    n = len(data)
    m = 1 + max(map(lambda m: max(max(m.header, default=0), max(m.body, default=0)), data))
    freq_matrix = np.zeros((n, 2 * m))

    for i in range(n):
        for j in data[i].header:
            freq_matrix[i][j] += 1

        for j in data[i].body:
            freq_matrix[i][m + j] += 1

    return csr_matrix(freq_matrix)


def tfidf_together(data: list[Message]) -> csr_matrix:
    n = len(data)
    vectorizer = TfidfVectorizer()
    raw_documents = []

    for i in range(n):
        document = data[i].header
        document.extend(data[i].body)
        document = " ".join(map(str, document))
        raw_documents.append(document)

    return csr_matrix(vectorizer.fit_transform(raw_documents))

def tfidf_separately(data: list[Message]) -> csr_matrix:
    n = len(data)
    vectorizer = TfidfVectorizer()
    raw_documents_1 = []
    raw_documents_2 = []

    for i in range(n):
        document_1 = " ".join(map(str, data[i].header))
        document_2 = " ".join(map(str, data[i].body))
        raw_documents_1.append(document_1)
        raw_documents_2.append(document_2)

    m_1 = csr_matrix(vectorizer.fit_transform(raw_documents_1))
    m_2 = csr_matrix(vectorizer.fit_transform(raw_documents_2))

    return hstack([m_1, m_2])
