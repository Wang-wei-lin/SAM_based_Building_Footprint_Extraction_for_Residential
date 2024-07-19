import laspy
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio


def las2shp(las_path, shp_path, crs_epsg="EPSG:32650"):
    """
    从.las格式的点云文件到.shp格式的矢量文件进行转换(Convert from .las point cloud files to .shp vector files)
    :param las_path: 点云文件路径(Point cloud file path)
    :param shp_path: 矢量文件的保存路径(path to save the vector file)
    :param crs_epsg: 坐标参考系统，默认为EPSG:32650(coordinate reference system, default is EPSG:32650)
    """
    las = laspy.read(las_path)

    x_list = las.x
    y_list = las.y
    z_list = las.z
    ri_list = las.intensity

    df = pd.DataFrame(data=np.array([x_list, y_list, z_list, ri_list]).T,columns=["X", "Y", "Z", "ref_int"])

    gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(x=x_list, y=y_list, crs=crs_epsg))

    gdf.to_file(shp_path)


def set_nodata_to_rgb_zero(input_path, output_path):
    """
    将图像的nodata值设为0(Set the image's nodata value to 0)
    :param input_path: 输入文件路径(Input file path)
    :param output_path: 输出文件路径(Output file path)
    """
    with rasterio.open(input_path) as src:

        meta = src.meta
        data = src.read()

        nodata = meta.get('nodata', None)

        if nodata is None:
            print("No nodata value found in metadata. Assuming nodata as a specific value is not possible.")
            return

        nodata_mask = (data[0] == nodata) | (data[1] == nodata) | (data[2] == nodata)

        data[0][nodata_mask] = 0
        data[1][nodata_mask] = 0
        data[2][nodata_mask] = 0

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)


def reduce_bit_depth(input_path, output_path):
    """
    将UTF-16格式的图像改为UTF-8格式(Convert UTF-16 images to UTF-8)
    :param input_path: 输入文件路径(Input file path)
    :param output_path: 输出文件路径(Output file path)
    """
    with rasterio.open(input_path) as src:
        # 读取图像数据（每个波段都是uint16类型）
        data = src.read()

        # 通过位操作保留每个像素值的低8位，即屏蔽掉高位字节
        data_8bit = data & 0xFF

        # 更新数据类型为uint8
        data_8bit = data_8bit.astype(np.uint8)

        # 获取并更新元数据
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, nodata=None)

        # 写入8位TIFF图像
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data_8bit)

