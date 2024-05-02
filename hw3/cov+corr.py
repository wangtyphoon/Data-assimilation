# %%
# 導入必要的函式庫
from netCDF4 import Dataset  # 用於讀取 NetCDF 格式檔案
import numpy as np  # 用於數值計算
import matplotlib.pyplot as plt  # 用於繪圖
import matplotlib.ticker as mticker  # 用於調整圖表刻度
import cartopy.crs as crs  # 用於地圖投影
from cartopy.feature import NaturalEarthFeature  # 用於添加地圖特徵

# %%
# 設定參數
ensLen = 42  # 集合成員數量
scaLen = 3  # 尺度數量 (大、中、小)
scaStr = 'LMS'  # 尺度的字串表示

# %%
# 開啟 NetCDF 檔案
ncfile = Dataset('./20180908_0400_d03_noda.nc')
# 讀取經緯度數據
xlon = np.array(ncfile['XLONG'])[0, :, :]
ylat = np.array(ncfile['XLAT'])[0, :, :]
yLen, xLen = xlon.shape  # 獲取經緯度數據的形狀

# 繪圖準備
clatlon = [np.sum(ylat)/ylat.size, np.sum(xlon)/xlon.size]  # 計算中心經緯度
cart_proj = crs.LambertConformal(central_longitude=clatlon[1], central_latitude=clatlon[0])  # 設定地圖投影
states = NaturalEarthFeature(category="cultural", scale="110m",
                           facecolor="none",
                           name="admin_1_states_provinces")  # 添加州省邊界

yxIdx = [(213, 196), (233, 205), (250, 205),
         (140, 20), (158, 76), (170, 121), (205, 149), (232, 175),
         (253, 240), (256, 275),
         (15, 255), (15, 230), (15, 205),
         (35, 255), (35, 230), (35, 205),
         (55, 255), (55, 230), (55, 205)]


uEnsMean = np.array(ncfile['U_mean_AtPlev925'])[0]  # 讀取 925 hPa 平均 U 風分量
vEnsMean = np.array(ncfile['V_mean_AtPlev925'])[0]  # 讀取 925 hPa 平均 V 風分量
windspeed = np.sqrt(uEnsMean**2+vEnsMean**2)  # 計算風速

uEnsPert = np.zeros([scaLen, ensLen, yLen, xLen])
vEnsPert = np.zeros([scaLen, ensLen, yLen, xLen])

for iens in range(ensLen):  # 遍歷所有集合成員
    for iSca in range(3):  # 遍歷所有尺度
        varName = 'EnsPert_ens'+'{:02d}'.format(iens+1)+'_sca'+scaStr[iSca]  # 構建變數名稱
        # 讀取 U 和 V 風擾動數據
        uEnsPert[iSca, iens] = np.array(ncfile['u'+varName])
        vEnsPert[iSca, iens] = np.array(ncfile['v'+varName])

uEnsSprd = np.sqrt(np.mean(np.sum(uEnsPert, axis=0)**2, axis=0))  # 計算 U 風集合擴散
valMax = np.ceil(np.max(uEnsSprd))  # 獲取集合擴散的最大值

# %% 計算cov 與 corr
uEnsPert_sum = np.sum(uEnsPert, axis=0)
vEnsPert_sum = np.sum(vEnsPert, axis=0)

def calculate_cov(reference_point, matrix):
    # 获取 reference_point 与每个点的协方差，并存储到一个空矩阵中
    cov_matrices = np.zeros((matrix.shape[1], matrix.shape[2]))

    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            compared_point = matrix[:, i, j]
            cov_matrix = np.cov(reference_point, compared_point)
            # 存储协方差值
            cov_matrices[i, j] = cov_matrix[0, 1]  # 取协方差矩阵中的第一个元素，即协方差值

    return cov_matrices

# 调用函数并获取结果
def draw_covariance(point_indices,reference_matrix, matrix,title):
    for i in point_indices:
        cov_matrix_result = calculate_cov(reference_matrix[:, yxIdx[i-1][0],yxIdx[i-1][1]], matrix)

        figh = plt.figure(figsize=(8, 7.5))
        ax = figh.add_axes([0.07, 0.0, 0.95, 1.0], projection=cart_proj)
        ax.add_feature(states, linewidth=0.5, edgecolor='black')
        ax.coastlines('50m', linewidth=0.8)

        yIdx, xIdx = yxIdx[i-1]
        iPt_lat = ylat[yIdx, xIdx]
        iPt_lon = xlon[yIdx, xIdx]
        ax.plot(iPt_lon, iPt_lat, 'ko', markersize=24, markerfacecolor='none', transform=crs.PlateCarree())
        ax.text(iPt_lon, iPt_lat, str(i), fontweight='heavy', va='center', ha='center', transform=crs.PlateCarree())

        # Create contour plot
        spread_contours = ax.contourf(xlon, ylat, cov_matrix_result, cmap='rainbow', extend='both', transform=crs.PlateCarree())

        # Add colorbar
        plt.colorbar(spread_contours, orientation='vertical', shrink=0.8, pad=0.02, label='Covariance')

        # Set title and labels
        plt.title(title+str(i))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig("cov/"+title+str(i)+".png")
        # Show the plot
        plt.show()

point_indices =[1,5,15]
title = "Covariance Matrix between Uwind and Uwind"
draw_covariance(point_indices,uEnsPert_sum,uEnsPert_sum,title )
title = "Covariance Matrix between Vwind and Vwind"
draw_covariance(point_indices,vEnsPert_sum,vEnsPert_sum,title )
title = "Covariance Matrix between Vwind and Uwind"
draw_covariance(point_indices,vEnsPert_sum,uEnsPert_sum,title )


def calculate_cov(reference_point, matrix):
    # 获取 reference_point 与每个点的协方差，并存储到一个空矩阵中
    cov_matrices = np.zeros((matrix.shape[1], matrix.shape[2]))

    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            compared_point = matrix[:, i, j]
            cov_matrix = np.corrcoef(reference_point, compared_point)
            # 存储协方差值
            cov_matrices[i, j] = cov_matrix[0, 1]  # 取协方差矩阵中的第一个元素，即协方差值
    return cov_matrices

# 调用函数并获取结果
def draw_correlation(point_indices,reference_matrix, matrix,title):
    for i in point_indices:
        cov_matrix_result = calculate_cov(reference_matrix[:, yxIdx[i-1][0],yxIdx[i-1][1]], matrix)

        figh = plt.figure(figsize=(8, 7.5))
        ax = figh.add_axes([0.07, 0.0, 0.95, 1.0], projection=cart_proj)
        ax.add_feature(states, linewidth=0.5, edgecolor='black')
        ax.coastlines('50m', linewidth=0.8)

        yIdx, xIdx = yxIdx[i-1]
        iPt_lat = ylat[yIdx, xIdx]
        iPt_lon = xlon[yIdx, xIdx]
        ax.plot(iPt_lon, iPt_lat, 'ko', markersize=24, markerfacecolor='none', transform=crs.PlateCarree())
        ax.text(iPt_lon, iPt_lat, str(i), fontweight='heavy', va='center', ha='center', transform=crs.PlateCarree())

        # Create contour plot
        spread_contours = ax.contourf(xlon, ylat, cov_matrix_result, cmap='rainbow', extend='both', transform=crs.PlateCarree())

        # Add colorbar
        plt.colorbar(spread_contours, orientation='vertical', shrink=0.8, pad=0.02, label='Covariance')

        # Set title and labels
        plt.title(title+str(i))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig("corr/"+title+str(i)+".png")
        # Show the plot
        plt.show()
title = "correlation coefficient between Uwind and Uwind"
draw_correlation(point_indices,uEnsPert_sum,uEnsPert_sum,title )
title = "correlation coefficient between Vwind and Vwind"
draw_correlation(point_indices,vEnsPert_sum,vEnsPert_sum,title )
title = "correlation coefficient between Vwind and Uwind"
draw_correlation(point_indices,vEnsPert_sum,uEnsPert_sum,title )