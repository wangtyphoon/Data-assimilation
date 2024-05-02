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

# %%
# 建立感興趣點的索引列表
yxIdx = [(213, 196), (233, 205), (250, 205),
         (140, 20), (158, 76), (170, 121), (205, 149), (232, 175),
         (253, 240), (256, 275),
         (15, 255), (15, 230), (15, 205),
         (35, 255), (35, 230), (35, 205),
         (55, 255), (55, 230), (55, 205)]
print("推薦點位:")
for iPt in range(len(yxIdx)):
    yIdx, xIdx = yxIdx[iPt]  # 獲取點位的 y 和 x 索引
    iPt_lat = ylat[yIdx, xIdx]  # 獲取點位的緯度
    iPt_lon = xlon[yIdx, xIdx]  # 獲取點位的經度
    # 打印點位信息
    print("xyIndex = ["+ '{:3d}'.format(xIdx)+ ', '+ '{:3d}'.format(yIdx)+']',
          '  ',
          'lon,lat = ['+'{:6.2f}'.format(iPt_lon)+'E, '+'{:5.2f}'.format(iPt_lat)+'N]')

# 定義一個函數，用於在地圖上繪製點位
def plot_points(ax):
    for iPt in range(len(yxIdx)):
        yIdx, xIdx = yxIdx[iPt]
        iPt_lat = ylat[yIdx, xIdx]
        iPt_lon = xlon[yIdx, xIdx]
        # 繪製點位，並標記點位編號
        ax.plot(iPt_lon, iPt_lat, 'ko', markersize=14, markerfacecolor='none', transform=crs.PlateCarree())
        ax.text(iPt_lon, iPt_lat, str(iPt+1), fontweight='heavy', va='center', ha='center', transform=crs.PlateCarree())


# %%
# 範例：繪製平均風場
uEnsMean = np.array(ncfile['U_mean_AtPlev925'])[0]  # 讀取 925 hPa 平均 U 風分量
vEnsMean = np.array(ncfile['V_mean_AtPlev925'])[0]  # 讀取 925 hPa 平均 V 風分量
windspeed = np.sqrt(uEnsMean**2+vEnsMean**2)  # 計算風速

figh = plt.figure(figsize=(8, 7.5))  # 建立圖形
ax = figh.add_axes([0.07, 0.0, 0.95, 1], projection=cart_proj)  # 添加子圖，並設定地圖投影
ax.add_feature(states, linewidth=0.5, edgecolor='black')  # 添加州省邊界
ax.coastlines('50m', linewidth=0.8)  # 添加海岸線

# 繪製風速等值線圖
wspd_contours = ax.contourf(xlon, ylat, windspeed,
                            levels=np.arange(1, 20+1, 1),
                            cmap='cubehelix_r',
                            extend='both',
                            transform=crs.PlateCarree())
cbar = plt.colorbar(wspd_contours, ax=ax, orientation='vertical', pad=.05)  # 添加顏色條
cbar.ax.tick_params(labelsize=14)
cbar.set_ticks(ticks=mticker.FixedLocator(np.arange(0, 20+1, 1*2)), update_ticks=True)  # 調整顏色條刻度

# 繪製風向箭頭
qvGap = 10  # 設定箭頭間距
ax.quiver(xlon[0::qvGap, 0::qvGap], ylat[0::qvGap, 0::qvGap],
          uEnsMean[0::qvGap, 0::qvGap], vEnsMean[0::qvGap, 0::qvGap],
          scale=25*20,
          transform=crs.PlateCarree())

# 繪製感興趣點位
plot_points(ax)

# 添加網格線
gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                  linestyle='--', draw_labels=True,
                  x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# 設定標題
ax.set_title("925 hPa 風場", fontsize=14)

# 顯示圖形
plt.show()

# %%
# 讀取尺度分離的風擾動數據
uEnsPert = np.zeros([scaLen, ensLen, yLen, xLen])
vEnsPert = np.zeros([scaLen, ensLen, yLen, xLen])

for iens in range(ensLen):  # 遍歷所有集合成員
    for iSca in range(3):  # 遍歷所有尺度
        varName = 'EnsPert_ens'+'{:02d}'.format(iens+1)+'_sca'+scaStr[iSca]  # 構建變數名稱
        # 讀取 U 和 V 風擾動數據
        uEnsPert[iSca, iens] = np.array(ncfile['u'+varName])
        vEnsPert[iSca, iens] = np.array(ncfile['v'+varName])

# %%
# 範例：計算並繪製集合擴散
uEnsSprd = np.sqrt(np.mean(np.sum(uEnsPert, axis=0)**2, axis=0))  # 計算 U 風集合擴散
valMax = np.ceil(np.max(uEnsSprd))  # 獲取集合擴散的最大值

figh = plt.figure(figsize=(8, 7.5))
ax = figh.add_axes([0.07, 0.0, 0.95, 1.0], projection=cart_proj)
ax.add_feature(states, linewidth=0.5, edgecolor='black')
ax.coastlines('50m', linewidth=0.8)

# 繪製集合擴散等值線圖
spread_contours = ax.contourf(xlon, ylat, uEnsSprd,
                              levels=np.arange(valMax/10, valMax*1.1, valMax/10),
                              cmap='cubehelix_r',
                              extend='both',
                              transform=crs.PlateCarree())

cbar = plt.colorbar(spread_contours, ax=ax, orientation='vertical', pad=.05)  # 添加顏色條
cbar.ax.tick_params(labelsize=14)

# 繪製感興趣點位
plot_points(ax)

# 添加網格線
gl = ax.gridlines(crs=crs.PlateCarree(), alpha=0.5,
                  linestyle='--', draw_labels=True,
                  x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# 設定標題
ax.set_title("925 hPa U 風集合擴散", fontsize=14)

# 儲存圖形
figh.savefig('./U_ensemble_spread.png', dpi=200)

# 顯示圖形
plt.show()

# 選擇三個點的索引
point_indices = [0, 4, 9]

# 獲取選定點的 U 和 V 風擾動時間序列
selected_u_pert = uEnsPert[:, :, yxIdx[point_indices[0]][0], yxIdx[point_indices[0]][1]]
selected_v_pert = vEnsPert[:, :, yxIdx[point_indices[1]][0], yxIdx[point_indices[1]][1]]

# 遍歷其他點，計算協方差和相關係數
# for i in range(len(yxIdx)):
#     if i not in point_indices:
#         print(i)
#         # 獲取其他點的 U 和 V 風擾動時間序列
#         other_u_pert = uEnsPert[:, :, yxIdx[i][0], yxIdx[i][1]]
#         other_v_pert = vEnsPert[:, :, yxIdx[i][0], yxIdx[i][1]]

#         # 計算協方差矩陣
#         cov_matrix = np.cov(np.concatenate((selected_u_pert, selected_v_pert), axis=0),
#                             np.concatenate((other_u_pert, other_v_pert), axis=0))

#         # 計算相關係數矩陣
#         corr_matrix = np.corrcoef(np.concatenate((selected_u_pert, selected_v_pert), axis=0),
#                                  np.concatenate((other_u_pert, other_v_pert), axis=0))

#         # 打印結果
#         print(f"點 {i+1} 與選定點的協方差矩陣：")
#         print(cov_matrix)
#         print(f"點 {i+1} 與選定點的相關係數矩陣：")
#         print(corr_matrix)

for i in point_indices:
    for j in range(len(yxIdx)):
        print(i)
        # 獲取其他點的 U 和 V 風擾動時間序列
        other_u_pert = uEnsPert[:, :, yxIdx[j][0], yxIdx[j][1]]
        other_v_pert = vEnsPert[:, :, yxIdx[j][0], yxIdx[j][1]]

        # 計算協方差矩陣
        cov_matrix = np.cov(np.concatenate((selected_u_pert, selected_v_pert), axis=0),
                            np.concatenate((other_u_pert, other_v_pert), axis=0))

        # 計算相關係數矩陣
        corr_matrix = np.corrcoef(np.concatenate((selected_u_pert, selected_v_pert), axis=0),
                                 np.concatenate((other_u_pert, other_v_pert), axis=0))

        # 打印結果
        print(f"點 {i+1} 與選定點的協方差矩陣：")
        print(cov_matrix)
        print(f"點 {i+1} 與選定點的相關係數矩陣：")
        print(corr_matrix)