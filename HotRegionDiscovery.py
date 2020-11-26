import pandas as pd
import numpy as np
from ast import literal_eval
from math import ceil
import gmplot
import os
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time


# lon 精度 ； lat 维度


class GridNode:
    def __init__(self, left, right, num):
        self.lat = right
        self.lon = left
        self.num = num


def Distance_euclidean(x, y):
    # 默认欧几里得距离
    assert len(x) == len(y)
    dis = 0
    for i in range(len(x)):
        dis += (x[i] - y[i]) ** 2
    return np.sqrt(dis)


class HotRegion(object):
    def __init__(self, delta, dataSet, noise_size, N_min):
        # delta 为密度阈值
        self.dataSet = dataSet
        self.delta = delta
        self.__Grid = None
        self.epsilon = None
        self.N_min = N_min
        self.noise_size = noise_size

    @property
    def Grid(self):
        return self.__Grid

    @Grid.setter
    def Grid(self, g):
        self.__Grid = g

    # 建立网格
    def Build_Grid(self):
        index = self.Find_lon_lat_index()
        print('网格上下限', index)

        grid = []
        # 设定网格规模
        self.epsilon = min(abs(index[0, 0] - index[0, 1]) / 100, abs(index[1, 0] - index[1, 1]) / 100)
        print('网格大小', self.epsilon)
        lon = index[0, 0] - 2*self.epsilon
        # 假设每个网格的中心点
        while lon < index[0, 1] + 2*self.epsilon:
            grid_pra = []
            lat = index[1, 0] - 2*self.epsilon
            while lat < index[1, 1] + 2*self.epsilon:
                Node = GridNode(lon, lat, 0)
                grid_pra.append(Node)
                lat += self.epsilon
            grid.append(grid_pra)
            lon += self.epsilon
        print('网格规模', len(grid), len(grid[0]))
        try:
            self.Grid = grid
            return True
        except:
            return False

    def Find_lon_lat_index(self):
        # Step 1
        dataSet = np.array(self.dataSet)
        lon_min = np.inf
        lon_max = -np.inf
        lat_min = np.inf
        lat_max = -np.inf
        for tra in dataSet:
            tra = np.array(tra)[:, 1:]
            for index in tra:
                if index[0] < lon_min:
                    lon_min = index[0]
                elif index[0] > lon_max:
                    lon_max = index[0]

                if index[1] < lat_min:
                    lat_min = index[1]
                elif index[1] > lat_max:
                    lat_max = index[1]

        return np.array([[lon_min, lon_max], [lat_min, lat_max]])

    # 更新网格
    # 简单外接矩形
    def Count_num_simple(self):
        for tra in self.dataSet:
            for i in range(len(tra)):
                _, lon_min, lat_min = np.min(tra[i:i + 2], axis=0)
                _, lon_max, lat_max = np.max(tra[i:i + 2], axis=0)
                index = [[], []]
                for vertical_coordinate in range(len(self.Grid)):
                    if lon_min <= self.Grid[vertical_coordinate][0].lon <= lon_max:
                        index[0].append(vertical_coordinate)
                for horizontal_coordinate in range(len(self.Grid[0])):
                    if lat_min <= self.Grid[0][horizontal_coordinate].lat <= lat_max:
                        index[1].append(horizontal_coordinate)
                for j in index[0]:
                    for k in index[1]:
                        self.Grid[j][k].num += 1
        print('矩形外延更新网格完成')

    # 更新网格-椭圆
    def Count_num_ellipse(self):
        for tra in self.dataSet:
            size = 0
            for i in range(len(tra) - 1):
                tra_A = tra[i][1:]
                tra_B = tra[i + 1][1:]
                distance_A_B = Distance_euclidean(tra_A, tra_B)
                # _为时间戳
                _, lon_min, lat_min = np.min(tra[i:i + 2], axis=0)
                _, lon_max, lat_max = np.max(tra[i:i + 2], axis=0)
                index = [[], []]
                for vertical_coordinate in range(len(self.Grid)):
                    if lon_min - 2 * self.noise_size <= self.Grid[vertical_coordinate][0].lon \
                            <= lon_max + 2 * self.noise_size:
                        index[0].append(vertical_coordinate)
                for horizontal_coordinate in range(len(self.Grid[0])):
                    if lat_min - 2 * self.noise_size <= self.Grid[0][horizontal_coordinate].lat \
                            <= lat_max + 2 * self.noise_size:
                        index[1].append(horizontal_coordinate)
                for j in index[0]:
                    for k in index[1]:
                        node_index = [self.Grid[j][k].lon, self.Grid[j][k].lat]
                        distance_count = (distance_A_B / 2) + self.noise_size  # 椭圆上的点到焦点的距离和
                        if Distance_euclidean(tra_A, node_index) + Distance_euclidean(tra_B, node_index) \
                                <= 2 * distance_count:
                            self.Grid[j][k].num += 1
        print('椭圆外延更新网格完成')

    # 更新网格-Bresenham
    def Count_num_Bresenham(self):
        for tra in self.dataSet:
            for i in range(len(tra) - 1):
                tra_A = tra[i][1:]
                tra_B = tra[i + 1][1:]

                # 在网格中定位tra-A，即获取tra-A所在网格的坐标
                index = []
                x_star = None
                x_end = None
                for vertical_coordinate in range(len(self.Grid)):
                    if tra_A[0] - self.epsilon/2 < self.Grid[vertical_coordinate][0].lon <= tra_A[0] + self.epsilon/2:
                        index.append(vertical_coordinate - 1)
                        x_star = self.Grid[vertical_coordinate-1][0].lon
                    if tra_B[0] - self.epsilon/2 <= self.Grid[vertical_coordinate][0].lon < tra_B[0] + self.epsilon/2:
                        x_end = self.Grid[vertical_coordinate-1][0].lon
                    if x_star and x_end:
                        break

                for horizontal_coordinate in range(len(self.Grid[0])):
                    if tra_A[1] <= self.Grid[0][horizontal_coordinate].lat:
                        index.append(horizontal_coordinate - 1)
                        break
                self.Grid[index[0]][index[1]].num += 1

                # x_star = tra_A[0]
                y_star = tra_A[1]
                # x_end = tra_B[0]
                y_end = tra_B[1]

                dif_x = x_end - x_star
                dif_y = y_end - y_star

                if abs(dif_x) >= abs(dif_y):
                    delta = dif_y / dif_x  # 斜率
                    # 此时x方向前进速度快
                    while True:
                        if abs(x_end - x_star) > self.epsilon:  # 判断x2与x1的距离与一个网格单位的大小关系
                            err = y_star - self.Grid[index[0]][index[1]].lat + self.epsilon/2  # 上一步点与横格栅的距离
                            delta_x = np.sign(x_end - x_star) * self.epsilon  # delta x
                            y_star += delta * delta_x  # 真实y
                            if self.Grid[index[0]][index[1]].lat < y_star < \
                                    self.Grid[index[0]][index[1]].lat + self.epsilon:  # 双向约束
                                # y方向未增加
                                index_new = [index[0] + int(np.sign(dif_x)), index[1]]
                                self.Grid[index_new[0]][index_new[1]].num += 1
                                index = index_new
                            else:
                                index_new = [index[0] + int(np.sign(dif_x)), index[1] + int(np.sign(dif_y))]
                                self.Grid[index_new[0]][index_new[1]].num += 1
                                err += y_star - self.Grid[index[0]][index[1]].lat - self.epsilon  # 加上下一步节点与下一个格子的横格栅的距离
                                if err < self.epsilon:  # 判断延直线往前进一个单位与网格单位的大小
                                    self.Grid[index_new[0] + int(np.sign(dif_x))][index_new[1]].num += 1
                                    index = index_new
                                elif err == self.epsilon:
                                    self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                    self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                    index = index_new
                                else:
                                    self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                    index = index_new
                            x_star += np.sign(dif_x)*self.epsilon

                        else:
                            # x前进至x_end
                            if self.Grid[index[0]][index[1]].lon < x_end < \
                                    self.Grid[index[0]][index[1]].lon + self.epsilon:
                                # 此时x_star与x_end在一个网格单元中
                                break
                            else:
                                if self.Grid[index[0]][index[1]].lat < y_end < \
                                        self.Grid[index[0]][index[1]].lat + self.epsilon:
                                    self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                    break
                                else:
                                    err = y_end - self.Grid[index[0]][index[1]].lat + self.epsilon + \
                                          y_star - self.Grid[index[0]][index[1]].lat
                                    index_new = [index[0] + int(np.sign(dif_x)), index[1] + int(np.sign(dif_y))]
                                    self.Grid[index_new[0]][index_new[1]].num += 1
                                    if err < self.epsilon:  # 判断延直线往前进一个单位与网格单位的大小
                                        self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                        break
                                    elif err == self.epsilon:
                                        self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                        self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                        break
                                    else:
                                        self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                        break
                else:
                    delta = dif_x / dif_y
                    # 此时y方向前进速度快
                    while True:
                        if abs(y_end - y_star) > self.epsilon:  # 判断y2与y1的距离与一个网格单元的大小关系
                            err = x_star - self.Grid[index[0]][index[1]].lon  # 上一步点与横格栅的距离
                            delta_y = np.sign(y_end - y_star) * self.epsilon
                            x_star += delta_y * delta
                            if self.Grid[index[0]][index[1]].lon < x_star < \
                                    self.Grid[index[0]][index[1]].lon + self.epsilon:
                                index_new = [index[0], index[1] + int(np.sign(dif_y))]
                                self.Grid[index_new[0]][index_new[1]].num += 1
                                index = index_new
                            else:
                                index_new = [index[0] + int(np.sign(dif_x)), index[1] + int(np.sign(dif_y))]
                                self.Grid[index_new[0]][index_new[1]].num += 1
                                err += x_star - self.Grid[index[0]][index[1]].lon - self.epsilon
                                if err < self.epsilon:  # 判断延直线往前进一个单位与网格单位的大小
                                    self.Grid[index_new[0]][index_new[1] + int(np.sign(dif_y))].num += 1
                                    index = index_new
                                elif err == self.epsilon:
                                    self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                    self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                    index = index_new
                                else:
                                    self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                    index = index_new
                            y_star += np.sign(dif_y)*self.epsilon
                        else:
                            if self.Grid[index[0]][index[1]].lat < y_end < self.Grid[index[0]][index[1]].lat + self.epsilon:
                                # 此时y_star与y_end在一个网格单元中
                                break
                            else:
                                if self.Grid[index[0]][index[1]].lon < x_end < self.Grid[index[0]][index[1]].lon + self.epsilon:
                                    self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                    break
                                else:
                                    err = x_end - self.Grid[index[0]][index[1]].lon + self.epsilon + \
                                          x_star - self.Grid[index[0]][index[1]].lon
                                    index_new = [index[0] + int(np.sign(dif_x)), index[1] + int(np.sign(dif_y))]
                                    self.Grid[index_new[0]][index[1]].num += 1
                                    if err < self.epsilon:
                                        self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                        break
                                    elif err == self.epsilon:
                                        self.Grid[index[0]][index[1] + int(np.sign(dif_y))].num += 1
                                        self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                        break
                                    else:
                                        self.Grid[index[0] + int(np.sign(dif_x))][index[1]].num += 1
                                        break

        print('基于Bresenham算法的全覆盖算法外延更新网格完成')

    def Updata_Grid(self, eps=None, min_sample=None):
        # 寻找热点网格点
        data = []
        index = []
        for i in range(len(self.Grid)):
            for j in range(len(self.Grid[0])):
                if self.Grid[i][j].num < self.N_min:
                    self.Grid[i][j].num = 0
                else:
                    data.append([self.Grid[i][j].lon, self.Grid[i][j].lat])
                    index.append((i, j))
        data = np.array(data)

        if not eps:
            eps = 2 * self.epsilon
        if not min_sample:
            min_sample = 6

        cluster_DBSCAN = DBSCAN(eps=eps, min_samples=min_sample)
        y_pred = cluster_DBSCAN.fit_predict(data)
        plt.scatter(data[:, 0], data[:, 1], c=y_pred)
        plt.show()

        # 寻找核心区域
        # for i in range(len(self.Grid)):
        #     for j in range(len(self.Grid[0])):
        #         if self.Grid[i][j].num != 0 and self.Grid[i - 1][j - 1].num != 0 and self.Grid[i - 1][j].num != 0 and \
        #             self.Grid[i - 1][j + 1].num != 0 and self.Grid[i][j - 1].num != 0 and self.Grid[i][j + 1].num != 0 \
        #             and self.Grid[i + 1][j - 1].num != 0 and self.Grid[i + 1][j].num != 0 and self.Grid[i + 1][j + 1].num != 0:
        #             data.append([self.Grid[i][j].lon, self.Grid[i][j].lat])
        #             index.append((i, j))
        #         else:
        #             self.Grid[i][j] = 0
        # print('核心区域更新完成')

        # 聚类
        # cluster_dbscan = DBSCAN(eps=)

    # 绘图
    def Print_All(self):
        # 注意绘图时，location=[lat， lon] 即维度在前，精度在后。
        path_dir = 'Maps_A2_LCSS'
        m = folium.Map(location=[53.350140, -6.266155], zoom_start=12)
        folium.Marker([53.350140, -6.266155]).add_to(m)
        gmap = gmplot.GoogleMapPlotter(53.350140, -6.266155, 12)
        i = 0
        for trs in self.dataSet:
            timestamp, lons, lats = zip(*trs)
            # print([[la, lo] for lo in lons for la in lats])
            folium.PolyLine(locations=[[lats[i], lons[i]] for i in range(len(lons))], color='green').add_to(m)
            gmap.plot(lats, lons, 'green', edge_width=3)
            i += 1
        gmap.draw(path_dir + os.sep + str(1) + '.html')
        m.save(path_dir + os.sep + str(2) + '.html')
        print(i)

    # 绘制热力图
    def Print_Hot(self, file):
        # file = 'Maps_A2_LCSS' + os.sep + 'Hot' + '.html'

        m = folium.Map(location=[53.350140, -6.266155], zoom_start=12)
        lat = np.array([[j.lat for j in i] for i in self.Grid]).flatten()
        lon = np.array([[j.lon for j in i] for i in self.Grid]).flatten()
        num = np.array([[j.num for j in i] for i in self.Grid]).flatten()
        # print(np.sum(num))

        data = np.array([[lat[i], lon[i], num[i]] for i in range(len(lat))])
        # print(data)
        HeatMap(data).add_to(m)
        m.save(file)


if __name__ == '__main__':
    # dataSet = pd.read_csv(r'D:\pycharm_workspace\Data_Structures_And_Algorithm\LCSS_dataset\test_set_a1.csv',
    #                       converters={"Trajectory": literal_eval},
    #                       sep='\t')
    # print(dataSet.head())

    # dataSet = pd.read_csv(r'D:\pycharm_workspace\Data_Structures_And_Algorithm\LCSS_dataset\train_set.csv',
    #                       converters={'Trajectory': literal_eval},
    #                       index_col='tripId')
    #
    # dataSet = np.array(dataSet['Trajectory'].tolist())
    #
    # for i in range(len(dataSet)):
    #     dataSet[i] = np.array(dataSet[i])
    #
    # print(dataSet[0][0, :], dataSet[0][40, :], dataSet[0][80, :])
    #
    # N = dataSet.shape[0]

    # for commissioning needs
    dataSet = np.array([[1.3539417e+15, -6.3195670e+00,  5.3299282e+01], [1.35394251e+15, -6.31703200e+00,  5.33160100e+01],
                        [1.3539433e+15, -6.2926540e+00,  5.3326519e+01]])

    G = HotRegion(dataSet=dataSet, delta=0.1, noise_size=0.0007, N_min=int(N / 4))
    # G.Build_Grid()
    # star = time.time()
    # G.Count_num_simple()
    # end = time.time()
    # print('简单矩形所用时间', end-star)
    # grid = []
    # for i in G.Grid:
    #     grid_pra = []
    #     for j in i:
    #         grid_pra.append(j.num)
    #     grid.append(grid_pra)
    # grid = np.array(grid)
    # print(np.sum(grid, axis=0))
    # file_simple = 'Maps_A2_LCSS' + os.sep + 'Hot' + '1' + '.html'
    # G.Print_Hot(file_simple)
    # G.Updata_Grid()

    file_ellipse = 'Maps_A2_LCSS' + os.sep + 'Hot' + '2' + '.html'
    G.Build_Grid()
    star = time.time()
    # G.Count_num_ellipse()
    G.Count_num_Bresenham()
    # G.Count_num_simple()
    end = time.time()
    print('所用时间', end - star)
    # G.Print_Hot(file_ellipse)
    # G.Updata_Grid()
    '''
    网格单元为500时
    简单矩形 测试集所用时间0.42119312286376953
    椭圆外延 测试集所用时间3.5706679821014404
    Bresenham 测试集所用时间0.7969040870666504
    
    网格单元为100时
    简单矩形 测试集所用时间0.07032656669616699
    椭圆外延 测试集所用时间0.33814120292663574
    Bresenham 测试集所用时间0.11271214485168457
    '''

