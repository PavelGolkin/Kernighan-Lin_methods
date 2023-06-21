import csv
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QRadioButton, \
    QTextEdit, QCheckBox


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация")
        self.resize(1280, 720)
        self.layout = QVBoxLayout()
        self.file_label = QLabel("Загрузка входного файла:")
        self.file_button = QPushButton("Обзор")
        self.file_button.clicked.connect(self.load_data)
        self.vertex_label = QLabel("Выбор алгоритма кластеризации:")
        self.radio_button1 = QRadioButton("Алгоритм Кернигана-Лина с выбором первой улучшающей разрез вершины при перестановке")
        self.radio_button2 = QRadioButton("Экспериментальный алгоритм Кернигана-Лина с условием наличия ребра между вершинами")
        self.radio_button3 = QRadioButton("Прямой алгоритм Кернигана-Лина на всех вершинах")
        self.radio_button4 = QRadioButton("Экспериментальный алгоритм Кернигана-Лина на всех вершинах с условием наличия ребра между вершинами")
        self.radio_button5 = QRadioButton("Алгоритм спектральной кластеризации")
        self.run_button = QPushButton("Запуск")
        self.run_button.clicked.connect(self.running)
        self.visualize_checkbox = QCheckBox("Визуализировать граф после кластеризации")
        self.output_label = QLabel("Вывод работы программы:")
        self.output_text = QTextEdit()



        self.layout.addWidget(self.file_label)
        self.layout.addWidget(self.file_button)
        self.layout.addWidget(self.vertex_label)
        self.layout.addWidget(self.radio_button1)
        self.layout.addWidget(self.radio_button2)
        self.layout.addWidget(self.radio_button3)
        self.layout.addWidget(self.radio_button4)
        self.layout.addWidget(self.radio_button5)
        self.layout.addWidget(self.visualize_checkbox)
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.output_text)
        self.setLayout(self.layout)

    def load_data(self):
        data = QFileDialog()
        data.setFileMode(QFileDialog.ExistingFile)
        data.setNameFilter("CSV Files (*.csv)")

        if data.exec_():
            choosed = data.selectedFiles()
            self.data_loaded = choosed[0]

    def running(self):

        try:
            if self.radio_button5.isChecked():
                graph_for_spectral = self.reading_sp(self.data_loaded)
            else:
                graph_for_KL = self.reading_CSV(self.data_loaded)


            if self.radio_button1.isChecked():
                choosing_algorithm = 1
            elif self.radio_button2.isChecked():
                choosing_algorithm = 2
            elif self.radio_button3.isChecked():
                choosing_algorithm = 3
            elif self.radio_button4.isChecked():
                choosing_algorithm = 4
            else :
                choosing_algorithm = 5


            if choosing_algorithm == 1:
                start_time = time.time()
                cluster1, cluster2, cut_size, iterations = self.Kernigan_Lin_first_better(graph_for_KL)
                work_time = time.time() - start_time
                cut = self.count_edges(graph_for_KL, cluster1, cluster2)

            elif choosing_algorithm == 2:
                start_time = time.time()
                cluster1, cluster2, cut_size, iterations = self.Kernigan_Lin_first_better_w_edge(graph_for_KL)
                work_time = time.time() - start_time
                cut = self.count_edges(graph_for_KL, cluster1, cluster2)

            elif choosing_algorithm == 3:
                start_time = time.time()
                cluster1, cluster2, cut_size, iterations = self.Kernigan_Lin_for_all_vertices(graph_for_KL)
                work_time = time.time() - start_time
                cut = self.count_edges(graph_for_KL, cluster1, cluster2)

            elif choosing_algorithm == 4:
                start_time = time.time()
                cluster1, cluster2, cut_size, iterations = self.Kernigan_Lin_for_all_vertices_w_edge(graph_for_KL)
                work_time = time.time() - start_time
                cut = self.count_edges(graph_for_KL, cluster1, cluster2)

            else:
                start_time = time.time()
                cluster1, cluster2 = self.spectral_algorithm(graph_for_spectral)
                work_time = time.time() - start_time

                iterations = " "
                cut = self.count_edges(graph_for_spectral, cluster1, cluster2)

            if self.visualize_checkbox.isChecked():
                if self.radio_button5.isChecked():
                    self.visualization(cluster1, cluster2, graph_for_spectral)
                else:
                    self.visualization(cluster1, cluster2, graph_for_KL)

            self.output_text.append("-----" + "Ошибки не найдены" + "-----")
            self.output_text.append("Первый кластер: " + str(cluster1))
            self.output_text.append("Второй кластер: " + str(cluster2))
            #self.output_text.append("Cut size: " + str(cut_size))
            self.output_text.append("Количество ребер между кластерами (разрез): " + str(cut))
            self.output_text.append("Количество итераций: " + str(iterations))
            self.output_text.append("Время выполнения алгоритма: " + str(work_time))
            self.output_text.append("-----"+"Программа успешно завершила свою работу"+"-----")
        except Exception as e:
            self.output_text.append("Работа программы прервана из-за ошибки: " + str(e))

    def visualization(self, A, B, graph):

        inp_data = nx.Graph(graph)
        position = nx.spring_layout(inp_data)

        for k in A:
            position[k][0] = -1 + position[k][0] / 2

        for k in B:
            position[k][0] = 0.5 + position[k][0] / 2

        nx.draw(inp_data, position, node_color=['red' if i in A else 'blue' for i in inp_data.nodes()])
        plt.show()

    def Kernigan_Lin_first_better(self, graph):
        def kL_algorithm(graph):
            n = len(graph)
            A = {v: v for v in range(n // 2)}
            B = {v: v for v in range(n // 2, n)}

            cut_size = checking_size(graph, A.keys(), B.keys())
            check = True
            step = 0

            while check:
                check = False
                for i in A.keys():
                    for j in B.keys():

                        # if graph[i][j] == 1:
                        bet = checking_bet(graph, i, j, A.values(), B.values())
                        if bet > 0:
                            A[i] = B[j]
                            B[j] = i
                            cut_size += bet
                            step += 1
                            check = True
                            break
                break

            return list(A.values()), list(B.values()), cut_size, step

        def checking_size(graph, A, B):
            size = 0
            for i in A:
                for j in B:
                    size += graph[i][j]
            return size

        def checking_bet(graph, i, j, A, B):
            bet = 0
            for k in A:
                bet -= graph[i][k]
                bet += graph[j][k]
            for k in B:
                bet -= graph[j][k]
                bet += graph[i][k]
            bet += graph[i][i]
            bet += graph[j][j]
            bet -= 2 * graph[i][j]
            return bet

        return kL_algorithm(graph)

    def Kernigan_Lin_first_better_w_edge(self, graph):
        def KL_alg_new(graph):
            lnn = len(graph)
            A = {v: v for v in range(lnn // 2)}
            B = {v: v for v in range(lnn // 2, lnn)}

            cut_size = checking_size(graph, A.keys(), B.keys())
            check = True
            step = 0

            while check:
                check = False
                for th in A.keys():
                    for j in B.keys():

                        if graph[th][j] == 1:
                            bet = checking_bet(graph, th, j, A.values(), B.values())

                            if bet > 0:
                                A[th] = B[j]
                                B[j] = th
                                cut_size += bet
                                step += 1
                                check = True
                                break
                break

            return list(A.values()), list(B.values()), cut_size, step

        def checking_size(graph, A, B):
            size = 0
            for i in A:
                for j in B:
                    size += graph[i][j]
            return size

        def checking_bet(graph, i, j, A, B):
            bet = 0
            for k in A:
                bet -= graph[i][k]
                bet += graph[j][k]
            for k in B:
                bet -= graph[j][k]
                bet += graph[i][k]
            bet += graph[i][i]
            bet += graph[j][j]
            bet -= 2 * graph[i][j]
            return bet

        return KL_alg_new(graph)

    def Kernigan_Lin_for_all_vertices(self, graph):
        def KL_all_vertices(graph):
            n = len(graph)

            A = {v: v for v in range(n // 2)}
            B = {v: v for v in range(n // 2, n)}

            cut_size = checking_size(graph, A.keys(), B.keys())
            check = True
            step = 0

            while check:
                check = False
                for i in A.keys():

                    best_better = 0
                    best_vert = None

                    for j in B.keys():
                        step += 1
                        bet = checking_bet(graph, i, j, A.values(), B.values())

                        if bet > best_better:
                            best_better = bet
                            best_vert = j

                    if best_vert is not None:
                        A[i] = B[best_vert]
                        B[best_vert] = i
                        cut_size += best_better
                        check = True

                return list(A.values()), list(B.values()), cut_size, step

        def checking_size(graph, A, B):
            size = 0

            for i in A:
                for j in B:

                    size += graph[i][j]

            return size

        def checking_bet(graph, i, j, A, B):
            bet = 0

            for k in A:
                bet -= graph[i][k]
                bet += graph[j][k]

            for k in B:
                bet -= graph[j][k]
                bet += graph[i][k]

            bet += graph[i][i]
            bet += graph[j][j]
            bet -= 2 * graph[i][j]

            return bet

        return KL_all_vertices(graph)

    def Kernigan_Lin_for_all_vertices_w_edge(self, graph):
        def KL_for_all_experimental(graph):
            n = len(graph)
            A = {v: v for v in range(n // 2)}
            B = {v: v for v in range(n // 2, n)}

            cut = checking_sze(graph, A.keys(), B.keys())
            chck = True
            step = 0

            while chck:
                chck = False
                for th in A.keys():
                    best = 0
                    best_vert = None

                    for ht in B.keys():

                        if graph[th][ht] == 1:
                            step += 1
                            bt = chcking_bet(graph, th, ht, A.values(), B.values())

                            if bt > best:
                                best = bt
                                best_vert = ht

                    if best_vert is not None:
                        A[th] = B[best_vert]
                        B[best_vert] = th
                        cut += best
                        chck = True

                return list(A.values()), list(B.values()), cut, step

        def checking_sze(graph, A, B):
            size = 0

            for i in A:
                for j in B:
                    size += graph[i][j]

            return size


        def chcking_bet(graph, i, j, A, B):
            bet = 0

            for k in A:
                bet -= graph[i][k]
                bet += graph[j][k]

            for k in B:
                bet -= graph[j][k]
                bet += graph[i][k]

            bet += graph[i][i]
            bet += graph[j][j]
            bet -= 2 * graph[i][j]

            return bet

        return KL_for_all_experimental(graph)


    def spectral_algorithm(self, graph):
        def spectral_clustering(adj_mat):
            normal = normalize(adj_mat, norm='l1', axis=0)

            eival, eivect = np.linalg.eig(normal)
            sorte = np.argsort(eival)
            sec_small_ei = eivect[:, sorte[1:3]]
            sec_small_ei = np.real(sec_small_ei)

            cl = KMeans(n_clusters=2, random_state=0)
            cl.fit(sec_small_ei)
            labels = cl.labels_

            return labels

        def cls_get(adjacency_matrix, labels):
            cl1 = []
            cl2 = []
            for i, label in enumerate(labels):
                if label == 0:
                    cl1.append(i)
                else:
                    cl2.append(i)
            return cl1, cl2

        lbl = spectral_clustering(graph)
        A, B = cls_get(graph, lbl)
        return A, B



    def count_edges(self, graph, cluster_a, cluster_b):
        rebr = 0

        for i in cluster_a:
            for j in cluster_b:

                if graph[i][j] == 1:
                    rebr += 1

        return rebr

    def reading_sp(self, file):
        data = {}

        with open(file, 'r') as file:
            for i in file:

                vert = i.strip().split(' ')
                n = int(vert[0])
                neighbor = [int(v) for v in vert[1:]]
                data[n] = neighbor

        num_n = len(data)
        ad_mat = [[0] * num_n for _ in range(num_n)]

        for n, neighbor in data.items():
            for neighbor in neighbor:

                ad_mat[n - 1][neighbor - 1] = 1
                ad_mat[neighbor - 1][n - 1] = 1

        return ad_mat

    def reading_CSV(self, file):
        with open(file) as file:
            read = csv.reader(file, delimiter=' ')
            graph = []

            for k in read:
                graph.append(list(map(int, k[1:])))

        n = len(graph)
        adj_matrix = np.zeros((n, n))

        for i in range(n):
            for j in graph[i]:
                adj_matrix[i][j - 1] = 1
                adj_matrix[j - 1][i] = 1

        return adj_matrix

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
